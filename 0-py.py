# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Init

# +
# import tensorflow as tf
import comet_ml
import gc
import multiprocessing as mp
import numpy as np
import torch
import os
import pytorch_lightning as pl
import spacy
import sentence_transformers
import torch.nn.functional as F
import torch.optim as optim
import shutil
import pandas as pd
import pyarrow.feather as feather

from collections import OrderedDict, defaultdict
from spacy.lang.en import English
from argparse import Namespace
from scipy.sparse import coo_matrix
from tqdm.auto import tqdm
from datetime import datetime
from operator import itemgetter
from sklearn import preprocessing
from sentence_transformers import SentenceTransformer
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data.dataset import random_split
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset, DataLoader

# working directory
ROOT_DIR = '.'
DATA_DIR = f'{ROOT_DIR}/data'
CHECKPOINT_DIR = 'd:/checkpoints/earnings-call'
CHECKPOINT_TEMP_DIR = f'{ROOT_DIR}/checkpoint/earnings-call/temp'

print(f'ROOT_DIR: {ROOT_DIR}')
print(f'DATA_DIR: {DATA_DIR}')
print(f'CHECKPOINT_DIR: {CHECKPOINT_DIR}')

# COMET API KEY
COMET_API_KEY = 'tOoHzzV1S039683RxEr2Hl9PX'

# set random seed
torch.backends.cudnn.deterministic = False;
torch.backends.cudnn.benchmark = True;
torch.backends.cudnn.enabled = True

# set device 'cuda' or 'cpu'
if torch.cuda.is_available():
    n_cuda = torch.cuda.device_count();
    
    def log_gpu_memory(verbose=False):
        torch.cuda.empty_cache()
        if verbose:
            for _ in range(n_cuda):
                print(f'GPU {_}:')
                print(f'{torch.cuda.memory_summary(_, abbreviated=True)}')
        else:
            for _ in range(n_cuda):
                memory_total = torch.cuda.get_device_properties(_).total_memory/(1024**3)
                memory_allocated = torch.cuda.memory_allocated(_)/(1024**3)
                print(f'GPU {_}: {memory_allocated: .2f}/{memory_total: .2f} (GB)')
            
    print(f'\n{n_cuda} GPUs found:');
    for _ in range(n_cuda):
        globals()[f'cuda{_}'] = torch.device(f'cuda:{_}');
        print(f'    {torch.cuda.get_device_name(_)} (cuda{_})');
        
    print('\nGPU memory:');
    log_gpu_memory();
else:
    print('GPU NOT enabled');
    
cpu = torch.device('cpu');
n_cpu = int(mp.cpu_count()/2);

print(f'\nCPU count (physical): {n_cpu}');


# -

# # Base

# ## helpers

# +
# helper: return current time
class Now:
    '''return current datetime, but has a more pretty format
    '''
    def __init__(self):
        self.current_datetime = datetime.now()
    def __repr__(self):
        return self.current_datetime.strftime('%H:%M:%S')  
    
# helper: refresh cuda memory
def refresh_cuda_memory():
    """
    Re-allocate all cuda memory to help alleviate fragmentation
    """
    # Run a full garbage collect first so any dangling tensors are released
    gc.collect()

    # Then move all tensors to the CPU
    for obj in gc.get_objects():
        if isinstance(obj, torch.Tensor) and obj.device!=cpu:
            obj.data = torch.empty(0)
            if isinstance(obj, torch.nn.Parameter) and obj.grad is not None:
                obj.grad.data = torch.empty(0)

    # Now empty the cache to flush the allocator
    torch.cuda.empty_cache()

# helper: flush chpt
def refresh_ckpt():
    '''
    move all `.ckpt` files to `/temp`
    '''
    # create ckpt_dir if not exists
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
    
    # create ckpt_temp_dir if not exists
    if not os.path.exists(CHECKPOINT_TEMP_DIR):
        os.makedirs(CHECKPOINT_TEMP_DIR)
    
    for name in os.listdir(CHECKPOINT_DIR):
        if name.endswith('.ckpt'):
            shutil.move(f'{CHECKPOINT_DIR}/{name}', f'{CHECKPOINT_DIR}/temp/{name}')

# helpers: load targets
def load_targets(targets_name):
    if 'targets_df' not in globals():
        print(f'Loading targets...@{Now()}')
        globals()['targets_df'] = pd.read_feather(f'{DATA_DIR}/{targets_name}.feather')
        print(f'Loading finished. @{Now()}')
        
# helpers: load preembeddings
def load_preembeddings(preembedding_type):
    if 'preembeddings' not in globals():
        print(f'Loading preembeddings...@{Now()}')
        globals()['preembeddings'] = torch.load(f"{DATA_DIR}/embeddings/preembeddings_{preembedding_type}.pt")
        print(f'Loading finished. @{Now()}')
        
# helpers: load split_df
def load_split_df(roll_type):
    split_df = pd.read_csv(f'{DATA_DIR}/split_dates.csv')
    globals()['split_df'] = split_df.loc[split_df.roll_type==roll_type]
    
# helper: log_ols_rmse
def log_ols_rmse(logger, window):
    '''
    Given window, find the corresponding ols_rmse from `bench_fr.feather`, 
    then log to Comet
    '''
    bench_fr = pd.read_feather('data/bench_fr.feather')

    ols_rmse_norm = bench_fr.loc[(bench_fr.roll_type=='3y') & (bench_fr.window==window)].test_rmse_fr_norm.to_list()[0]
    logger.experiment.log_parameter('ols_rmse_norm', ols_rmse_norm)
    
def log_test_start(logger, roll_type, window):
    '''
    Given window, find the corresponding ols_rmse from `bench_fr.feather`, 
    then log to Comet
    '''
    split_df = pd.read_csv(f'data/split_dates.csv')

    _, train_start, train_end, test_start, test_end, _ = tuple(split_df.loc[(split_df.window==window) & (split_df.roll_type==roll_type)].iloc[0])
    
    logger.experiment.log_parameter('train_start', train_start)
    logger.experiment.log_parameter('train_end', train_end)
    logger.experiment.log_parameter('test_start', test_start)
    logger.experiment.log_parameter('test_end', test_end)


# -

# ## `train`

# loop one
def train_one(Model, window_i, model_hparams, train_hparams):
    
    # set window
    model_hparams.update({'window': split_df.iloc[window_i].window})
    
    # init model
    model = Model(model_hparams)

    # get model type
    model_hparams['model_type'] = model.model_type
    model_hparams['target_type'] = model.target_type
    model_hparams['feature_type'] = model.feature_type
    model_hparams['normalize_target'] = model.normalize_target
    model_hparams['attn_type'] = model.attn_type
    if hasattr(model, 'emb_share'):
        model_hparams['emb_share'] = model.emb_share
    
    # checkpoint
    ckpt_prefix = f"{train_hparams['note']}_{model_hparams['window']}_".replace('*',  '')
    
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        verbose=False,
        mode='min',
        monitor='val_loss',
        filepath=CHECKPOINT_DIR,
        prefix=ckpt_prefix,
        save_top_k=train_hparams['save_top_k'],
        period=train_hparams['checkpoint_period'])

    # logger
    logger = pl.loggers.CometLogger(
        api_key=COMET_API_KEY,
        save_dir='/data/logs',
        project_name='earnings-call',
        experiment_name=model_hparams['window'],
        workspace='amiao',
        display_summary_level=0)

    # early stop
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=train_hparams['early_stop_patience'],
        verbose=False,
        mode='min')

    # trainer
    trainer = pl.Trainer(gpus=model_hparams['gpus'], 
                         precision=train_hparams['precision'],
                         checkpoint_callback=checkpoint_callback, 
                         early_stop_callback=early_stop_callback,
                         overfit_batches=train_hparams['overfit_batches'], 
                         row_log_interval=train_hparams['row_log_interval'],
                         val_check_interval=train_hparams['val_check_interval'], 
                         progress_bar_refresh_rate=1, 
                         distributed_backend='dp', 
                         accumulate_grad_batches=train_hparams['accumulate_grad_batches'],
                         min_epochs=train_hparams['min_epochs'],
                         max_epochs=train_hparams['max_epochs'], 
                         max_steps=train_hparams['max_steps'], 
                         logger=logger)

    # delete unused hparam
    if model.model_type=='mlp': model_hparams.pop('final_tdim',None)
    if model.feature_type=='fin-ratio': 
        model_hparams.pop('max_seq_len',None)
        model_hparams.pop('n_layers_encoder',None)
        model_hparams.pop('n_head_encoder',None)
        model_hparams.pop('d_model',None)
        model_hparams.pop('dff',None)
    if model.feature_type=='text': 
        model_hparams.pop('normalize_layer',None)
        model_hparams.pop('normalize_batch',None)
    if model.attn_type!='mha': model_hparams.pop('n_head_decoder',None)

    # add n_model_params
    train_hparams['n_model_params'] = sum(p.numel() for p in model.parameters())

    # upload hparams
    logger.experiment.log_parameters(model_hparams)
    logger.experiment.log_parameters(train_hparams)
    
    # upload ols_rmse (for reference)
    log_ols_rmse(logger, model_hparams['window'])
    
    # upload test_start
    log_test_start(logger, model_hparams['roll_type'], model_hparams['window'])
    
    # If run on ASU, upload code explicitly
    if train_hparams['machine'] == 'ASU':
        codefile = [name for name in os.listdir('.') if name.endswith('.py')]
        assert len(codefile)==1, f'There must be only one `.py` file in the current directory! {len(codefile)} files detected: {codefile}'
        logger.experiment.log_asset(codefile[0])
    
    
    # refresh GPU memory
    refresh_cuda_memory()

    # fit and test
    try:
        # train the model
        trainer.fit(model)

        # test on the best model
        trainer.test(ckpt_path='best')

    except RuntimeError as e:
        raise e
    finally:
        del model, trainer
        refresh_cuda_memory()
        logger.finalize('finished')


# ## `Dataset`

# Dataset: Txt + Fin-ratio
class CCDataset(Dataset):
    
    def __init__(self, split_window, split_type, text_in_dataset, roll_type, print_window, preembeddings, targets_df, split_df, gpus, valid_transcriptids=None):
        '''
        Args:
            preembeddings (from globals): list of embeddings. Each element is a tensor (S, E) where S is number of sentences in a call
            targets_df (from globals): DataFrame of targets variables.
            split_df (from globals):
            split_window: str. e.g., "roll-09"
            split_type: str. 'train' or 'test'
            text_only: only output CAR and transcripts if true, otherwise also output financial ratios
            transcriptids: list. If provided, only the given transcripts will be used in generating the Dataset. `transcriptids` is applied **on top of** `split_window` and `split_type`
            gpus: gpus used, should be a list. ex, [0,1]
        '''

        # get split dates from `split_df`
        _, train_start, train_end, test_start, test_end, _ = tuple(split_df.loc[(split_df.window==split_window) & (split_df.roll_type==roll_type)].iloc[0])
        # print current window
        if print_window:
            print(f'Current window: {split_window} ({roll_type}) \n(train: {train_start} to {train_end}) (test: {test_start} to {test_end})')
        
        train_start = datetime.strptime(train_start, '%Y-%m-%d').date()
        train_end = datetime.strptime(train_end, '%Y-%m-%d').date()
        test_start = datetime.strptime(test_start, '%Y-%m-%d').date()
        test_end = datetime.strptime(test_end, '%Y-%m-%d').date()
        
        # select valid transcriptids (preemb_keys) according to split dates 
        if split_type=='train':
            transcriptids = targets_df[targets_df.ciq_call_date.between(train_start, train_end)].transcriptid.sample(frac=1, random_state=42).tolist()
            transcriptids = transcriptids[:int(len(transcriptids)*0.9)]
            # print(f'Dateset -> N train: {len(transcriptids)}')
            
        if split_type=='val':
            transcriptids = targets_df[targets_df.ciq_call_date.between(train_start, train_end)].transcriptid.sample(frac=1, random_state=42).tolist()
            transcriptids = transcriptids[int(len(transcriptids)*0.9):]
            # print(f'Dataset -> N val: {len(transcriptids)}')

        elif split_type=='test':
            transcriptids = targets_df[targets_df.ciq_call_date.between(test_start, test_end)].transcriptid.tolist()
            # print(f'Dataset -> N test: {len(transcriptids)}')


        self.valid_preemb_keys = set(transcriptids).intersection(set(preembeddings.keys()))
        
        if valid_transcriptids is not None:
            self.valid_preemb_keys = self.valid_preemb_keys.intersection(set(valid_transcriptids))
            
        # remove last few samples from `valid_preemb_keys` so that it's divisible by the number of gpus
        if split_type in ['train', 'val']:
            max_valid_preemb_keys_len = len(self.valid_preemb_keys)//len(gpus)*len(gpus)
            self.valid_preemb_keys = list(self.valid_preemb_keys)[:(max_valid_preemb_keys_len)]
        
        # self attributes
        self.text_in_dataset = text_in_dataset
        if text_in_dataset:
            self.preembeddings = preembeddings
        self.targets_df = targets_df
        self.sent_len = sorted([(k, preembeddings[k].shape[0]) for k in self.valid_preemb_keys], key=itemgetter(1))
        self.train_start = train_start
        self.train_end = train_end
        self.test_start = test_start
        self.test_end = test_end
        self.n_samples = len(self.sent_len)
        self.split_window = split_window
        self.split_type = split_type
        
    def __len__(self):
        return (len(self.valid_preemb_keys))
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        transcriptid = self.sent_len[idx][0]
        targets = self.targets_df[self.targets_df.transcriptid==transcriptid].iloc[0]
        
        # all of the following targests are
        # of type `numpy.float64`
        docid = targets.docid
        
        sue = targets.sue_norm
        sest = targets.sest_norm
        car_0_30 = targets.car_0_30
        car_0_30_norm = targets.car_0_30_norm
        revision = targets.revision
        revision_norm = targets.revision_norm
        inflow = targets.inflow
        inflow_norm = targets.inflow_norm
        
        similarity = targets.similarity_bigram_norm
        sentiment = targets.qa_positive_sent_norm
        
        
        alpha = targets.alpha_norm
        volatility = targets.volatility_norm
        mcap = targets.mcap_norm
        bm = targets.bm_norm
        roa = targets.roa_norm
        debt_asset = targets.debt_asset_norm
        numest = targets.numest_norm
        smedest = targets.smedest_norm
        sstdest = targets.sstdest_norm
        car_m1_m1 = targets.car_m1_m1_norm
        car_m2_m2 = targets.car_m2_m2_norm
        car_m30_m3 = targets.car_m30_m3_norm
        volume = targets.volume_norm
        
        if self.text_in_dataset:
            # inputs: preembeddings
            embeddings = self.preembeddings[transcriptid]
            
            return car_0_30, car_0_30_norm, inflow, inflow_norm, revision, revision_norm, \
                   transcriptid, embeddings, \
                   [alpha, car_m1_m1, car_m2_m2, car_m30_m3, sest, sue, numest, sstdest, smedest, mcap, roa, bm, debt_asset, volatility, volume]
        else:
            return docid, \
                   torch.tensor(car_0_30,dtype=torch.float32), \
                   torch.tensor(car_0_30_norm,dtype=torch.float32), \
                   torch.tensor([similarity, sentiment],dtype=torch.float32),\
                   torch.tensor([alpha, car_m1_m1, car_m2_m2, car_m30_m3, sest, sue, numest, sstdest, smedest, mcap, roa, bm, debt_asset, volatility, volume], dtype=torch.float32)


# ## `Model`

# +
# Model: position encoder
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # pe: (max_len, 1, d_model)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :] # (S, N, E)
        return self.dropout(x)
    
# Model: Base
class CC(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        
        # self.text_in_dataset will be filled during instanciating.
        self.hparams = Namespace(**hparams)
        
        # check: batch_size//len(gpus)
        assert self.hparams.batch_size%len(self.hparams.gpus)==0, \
            f'`batch_size` must be divisible by `len(gpus)`. Currently batch_size={self.hparams.batch_size}, gpus={self.hparams.gpus}'
        # check: val_batch_size//len(gpus)
        assert self.hparams.val_batch_size%len(self.hparams.gpus)==0, \
            f'`val_batch_size` must be divisible by `len(gpus)`. Currently batch_size={self.hparams.val_batch_size}, gpus={self.hparams.gpus}'
        
        global preembeddings, targets_df, split_df
        self.preembeddings = preembeddings
        self.targets_df = targets_df
        self.split_df = split_df

    # forward
    def forward(self):
        pass
    
    # loss
    def mse_loss(self, y, t):
        return F.mse_loss(y, t)
        
    # validation step
    def validation_epoch_end(self, outputs):
        mse = torch.stack([x['val_loss'] for x in outputs]).mean()
        rmse = torch.sqrt(mse)
        
        log_dict = {'val_rmse': rmse}
        
        if 'val_loss_car' in outputs[0]:
            rmse_car = torch.sqrt(torch.stack([x['val_loss_car'] for x in outputs]).mean())
            log_dict['val_rmse_car'] = rmse_car
            
        if 'val_loss_inflow' in outputs[0]:
            rmse_inflow = torch.sqrt(torch.stack([x['val_loss_inflow'] for x in outputs]).mean())
            log_dict['val_rmse_inflow'] = rmse_inflow

        if 'val_loss_revision' in outputs[0]:
            rmse_revision = torch.sqrt(torch.stack([x['val_loss_revision'] for x in outputs]).mean())
            log_dict['val_rmse_revision'] = rmse_revision

        return {'val_loss': mse, 'log': log_dict}
    
    # test step
    def test_epoch_end(self, outputs):
        mse = torch.stack([x['test_loss'] for x in outputs]).mean()
        rmse = torch.sqrt(mse)
        
        log_dict = {'test_rmse': rmse}
        
        if 'test_loss_car' in outputs[0]:
            rmse_car = torch.sqrt(torch.stack([x['test_loss_car'] for x in outputs]).mean())
            log_dict['test_rmse_car'] = rmse_car

        if 'test_loss_inflow' in outputs[0]:
            rmse_inflow = torch.sqrt(torch.stack([x['test_loss_inflow'] for x in outputs]).mean())
            log_dict['test_rmse_inflow'] = rmse_inflow
            
        if 'test_loss_revision' in outputs[0]:
            rmse_revision = torch.sqrt(torch.stack([x['test_loss_revision'] for x in outputs]).mean())
            log_dict['test_rmse_revision'] = rmse_revision
            
        return {'test_loss': mse, 'log': log_dict, 'progress_bar':log_dict}
    
    # Dataset
    def prepare_data(self):
        
        self.train_dataset = CCDataset(self.hparams.window, split_type='train', text_in_dataset=self.text_in_dataset,
                                       roll_type=self.hparams.roll_type, print_window=True, preembeddings=self.preembeddings,
                                       targets_df=self.targets_df, split_df=self.split_df, gpus=self.hparams.gpus)
        print(f'N train = {len(self.train_dataset)}')
        
        self.val_dataset = CCDataset(self.hparams.window, split_type='val', text_in_dataset=self.text_in_dataset,
                                     roll_type=self.hparams.roll_type, print_window=False, preembeddings=self.preembeddings,
                                     targets_df=self.targets_df, split_df=self.split_df, gpus=self.hparams.gpus)
        print(f'N val = {len(self.val_dataset)}')
        print(f'N train+val = {len(self.train_dataset)+len(self.val_dataset)}')

        self.test_dataset = CCDataset(self.hparams.window, split_type='test', text_in_dataset=self.text_in_dataset, 
                                      roll_type=self.hparams.roll_type, print_window=False, preembeddings=self.preembeddings,
                                      targets_df=self.targets_df, split_df=self.split_df, gpus=self.hparams.gpus)
        print(f'N test = {len(self.test_dataset)}')

    # DataLoader
    def train_dataloader(self):
        # Caution:
        # - If you enable `BatchNorm`, then must set `drop_last=True`.

        collate_fn = self.collate_fn if self.text_in_dataset else None
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True, drop_last=False, num_workers=0, pin_memory=True, collate_fn=collate_fn)
    
    def val_dataloader(self):
        # Caution: 
        # - To improve the validation speed, I'll set val_batch_size to 4. 
        # - Must set `drop_last=True`, otherwise the `val_loss` tensors for different batches won't match and hence give you error.
        # - Not to set `val_batch_size` too large (e.g., 16), otherwise you'll lose precious validation data points
        
        collate_fn = self.collate_fn if self.text_in_dataset else None
        return DataLoader(self.val_dataset, batch_size=self.hparams.val_batch_size, num_workers=0, pin_memory=True, collate_fn=collate_fn, drop_last=False)

    def test_dataloader(self):
        collate_fn = self.collate_fn if self.text_in_dataset else None
        return DataLoader(self.test_dataset, num_workers=0, pin_memory=True, collate_fn=collate_fn, drop_last=False)
    
    def collate_fn(self, data):
        '''create mini-batch

        Retures:
            embeddings: tensor, (N, S, E)
            mask: tensor, (N, S)
            sue,car,selead,sest: tensor, (N,)
        '''
        
        # embeddings: (N, S, E)
        car_0_30, car_0_30_norm, inflow, inflow_norm, revision, revision_norm, \
        transcriptid, embeddings, \
        fin_ratios = zip(*data)
        
        # pad sequence
        # the number of `padding_value` is irrelevant, since we'll 
        # apply a mask in the Transformer encoder, which will 
        # eliminate the padded positions.
        valid_seq_len = [emb.shape[-2] for emb in embeddings]
        embeddings = pad_sequence(embeddings, batch_first=True, padding_value=0) # (N, T, E)

        # mask: (N, T)
        mask = torch.ones((embeddings.shape[0], embeddings.shape[1]))
        for i, length in enumerate(valid_seq_len):
            mask[i, :length] = 0
        mask = mask == 1
        
        return torch.tensor(car_0_30, dtype=torch.float32), torch.tensor(car_0_30_norm, dtype=torch.float32), \
               torch.tensor(inflow, dtype=torch.float32), torch.tensor(inflow_norm, dtype=torch.float32), \
               torch.tensor(revision, dtype=torch.float32), torch.tensor(revision_norm, dtype=torch.float32), \
               torch.tensor(transcriptid, dtype=torch.float32), embeddings.float(), mask, \
               torch.tensor(fin_ratios, dtype=torch.float32)
        
    # optimizer
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer   


# -

# # MLP

# ## model

# MLP
class CCMLP(CC):
    def __init__(self, hparams):
        # by super().__init__, `self.hparams` will be created
        super().__init__(hparams)
        
        # attibutes
        self.model_type = 'MLP'
        self.target_type = 'car'
        self.feature_type = 'fr+mtxt'
        self.normalize_target = True
        self.attn_type = 'dotprod'
        
        self.text_in_dataset = True if self.feature_type not in ['fr', 'fr+mtxt'] else False 
        
        # dropout layers
        self.dropout_1 = nn.Dropout(self.hparams.dropout)
        self.dropout_2 = nn.Dropout(self.hparams.dropout)
        self.dropout_3 = nn.Dropout(self.hparams.dropout)
        
        # fc layers
        self.fc_1 = nn.Linear(17, 32)
        self.fc_2 = nn.Linear(32, 32)
        self.fc_3 = nn.Linear(32, 32)
        self.fc_4 = nn.Linear(32, 32)
        self.fc_5 = nn.Linear(32, 1)
        
    # forward
    def forward(self, fin_ratios, manual_txt):
        x = torch.cat([fin_ratios, manual_txt], dim=-1) # (N, 2+15)

        x_car = F.relu(self.fc_1(x))
        x_car = F.relu(self.fc_2(x_car))
        x_car = F.relu(self.fc_3(x_car))
        x_car = F.relu(self.fc_4(x_car))
        y_car = self.fc_5(x_car) # (N, 1)
        
        return y_car
    
        
    # train step
    def training_step(self, batch, idx):
        
        _, car, car_norm, manual_txt, fin_ratio = batch
        
        # forward
        y_car = self.forward(fin_ratio, manual_txt) # (N, 1)

        # compute loss
        loss_car = self.mse_loss(y_car, car_norm.unsqueeze(-1)) # ()
        
        # logging
        return {'loss': loss_car, 'log': {'train_loss': loss_car}}
        
    # validation step
    def validation_step(self, batch, idx):
        
        _, car, car_norm, manual_txt, fin_ratio = batch
        
        # forward
        y_car = self.forward(fin_ratio, manual_txt) # (N, 1)

        # compute loss
        loss_car = self.mse_loss(y_car, car_norm.unsqueeze(-1)) # ()
        
        # logging
        return {'val_loss': loss_car}    
        
    # test step
    def test_step(self, batch, idx):
        
        _, car, car_norm, manual_txt, fin_ratio = batch
        
        # forward
        y_car = self.forward(fin_ratio, manual_txt) # (N, 1)

        # compute loss
        loss_car = self.mse_loss(y_car, car_norm.unsqueeze(-1)) # ()
        
        # logging
        return {'test_loss': loss_car}  

# ## run

# +
# choose Model
Model = CCMLP

# model hparams
model_hparams = {
    'preembedding_type': 'all_sbert_roberta_nlistsb_encoded', # key!
    'targets_name': 'f_sue_keydevid_car_finratio_vol_transcriptid_sim_inflow_revision_sentiment_text_norm', # key!
    'gpus': [0,1], # key
    'seed': 42,    # key
    'roll_type': '5y',
    'batch_size': 128,
    'val_batch_size':128,
    'learning_rate': 5e-4,
    'task_weight': 1,
    'dropout': 0.5,
}

# train hparams
train_hparams = {
    # checkpoint & log
    'machine': 'yu-workstation', # key!
    'note': f"MLP-14,(car~fr+mtxt {model_hparams['roll_type']}),hidden=32,hiddenLayer=4,fc_dropout=no,NormCAR=yes,bsz={model_hparams['batch_size']},seed={model_hparams['seed']},log(mcap)=yes,lr={model_hparams['learning_rate']:.2g}", # key!
    'row_log_interval': 10,
    'save_top_k': 1,
    'val_check_interval': 1.0,

    # data size
    'precision': 32, # key!
    'overfit_batches': 0.0,
    'min_epochs': 10, 
    'max_epochs': 50, 
    'max_steps': None,
    'accumulate_grad_batches': 1,

    # Caution:
    # The check of patience depends on **how often you compute your val_loss** (`val_check_interval`). 
    # Say you check val every N baches, then `early_stop_callback` will compare to your latest N **baches**.
    # If you compute val_loss every N **epoches**, then `early_stop_callback` will compare to the latest N **epochs**.
    'early_stop_patience': 6,

    # Caution:
    # If set to 1, then save ckpt every 1 epoch
    # If set to 0, then save ckpt on every val!!! (if val improves)
    'checkpoint_period': 1}

# delete all existing .ckpt files
refresh_ckpt()

# load split_df
load_split_df(model_hparams['roll_type'])
    
# load targets_df
load_targets(model_hparams['targets_name'])

# load preembeddings
# you have to do this because CCDataset requires it
load_preembeddings(model_hparams['preembedding_type'])
    
# loop over 24!
np.random.seed(model_hparams['seed'])
torch.manual_seed(model_hparams['seed'])

for window_i in range(len(split_df)):

    # train one window
    train_one(Model, window_i, model_hparams, train_hparams)
