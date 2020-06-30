# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.5.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Init

# %%
# import modules
import collections
import functools
import pickle
import platform
import random
import re
import gc
import os
import requests
import scipy
from datetime import datetime
from functools import partial
from operator import itemgetter
import comet_ml
from torch import nn
import multiprocessing as mp

import pandas as pd
import numpy as np
import comet_ml
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
import shutil
import sklearn
import torch
from collections import OrderedDict, defaultdict
from argparse import Namespace
from scipy.sparse import coo_matrix
from tqdm.auto import trange, tqdm
from torch.utils.data import Dataset, DataLoader

from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data.dataset import random_split
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

# helper functions
class Now:
    '''return current datetime, but has a more pretty format
    '''
    def __init__(self):
        self.current_datetime = datetime.now()
    def __repr__(self):
        return self.current_datetime.strftime('%H:%M:%S')    

def unique(seq: 'tuple or list') -> list:
    '''remove duplicate while keeping original order
    '''
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

def uniqueN(seq: 'tuple or list') -> list:
    '''remove duplicate while keeping original order
    '''
    return len(unique(seq))

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def sv(objname: str, savename: str = None, path='./data', log=True):
    '''
    log: output success messsage at the end
    '''
    assert isinstance(objname, str)
    if savename is None:
        savename = objname
    save_path = f"{path}/{savename}.pkl"
    obj = globals()[objname]
    with open(save_path, 'wb') as f:
        pickle.dump(obj, f)
    if log is True:
        if savename == objname:
            print(f'-{objname}- saved')
        else:
            print(f'-{objname}- saved as -{savename}-')
    
def ld(filename: str, objname=None, path='./data', force=False, log=True):
    assert isinstance(filename, str)
    load_path = f"{path}/{filename}.pkl"
    if objname is None:
        objname = filename
    
    # create objname in globals()
    if_exists = objname in globals()
    if if_exists == False:
        with open(load_path, 'rb') as f:
            varval = pickle.load(f)
        globals()[objname] = varval
        if log is True:
            if filename == objname:
                print(f'-{filename}- loaded')
            else:
                print(f'-{filename}- loaded as -{objname}-')
    elif if_exists == True:
        if force==False:
            print(f'-{objname}- already exists, will not load again!')
        elif force==True:
            with open(load_path, 'rb') as f:
                varval = pickle.load(f)
            globals()[objname] = varval
            print(f'-{filename}- loaded as -{objname}- (forced)')

# working directory
ROOT_DIR = '.'
DATA_DIR = f'{ROOT_DIR}/data'
CHECKPOINT_DIR = f'{ROOT_DIR}/checkpoint/earnings-call'
CHECKPOINT_TEMP_DIR = f'{ROOT_DIR}/checkpoint/earnings-call/temp'
print(f'ROOT_DIR: {ROOT_DIR}')
print(f'DATA_DIR: {DATA_DIR}')
print(f'CHECKPOINT_DIR: {CHECKPOINT_DIR}')

# Comet API key
COMET_API_KEY = 'tOoHzzV1S039683RxEr2Hl9PX'

# set random seed
np.random.seed(42)
torch.manual_seed(42);
torch.backends.cudnn.deterministic = False;
torch.backends.cudnn.benchmark = True;

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


# %% [markdown]
# # Base

# %% [markdown]
# ## helpers

# %%
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
def refresh_ckpt(ckpt_path):
    '''
    move all `.ckpt` files to `/temp`
    '''
    # create ckpt temp if not exist
    if not os.path.exists(f'{ckpt_path}/temp/'):
        os.makedirs(f'{ckpt_path}/temp/')
    for name in os.listdir(ckpt_path):
        if name.endswith('.ckpt'):
            shutil.move(f'{ckpt_path}/{name}', f'{ckpt_path}/temp/{name}')

# helpers: load targets
def load_targets(targets_name):
    if 'targets_df' not in globals():
        globals()['targets_df'] = pd.read_feather(f'{DATA_DIR}/{targets_name}.feather')
        
# helpers: load preembeddings
def load_preembeddings(preembedding_type):
    if 'preembeddings' not in globals():
        print(f'Loading preembeddings...@{Now()}')
        globals()['preembeddings'] = torch.load(f"{DATA_DIR}/preembeddings_{preembedding_type}.pt")
        print(f'Loading finished. {Now()}')
        
# helpers: load split_df
def load_split_df(roll_type):
    split_df = pd.read_csv(f'{DATA_DIR}/split_dates.csv')
    globals()['split_df'] = split_df.loc[split_df.roll_type==roll_type]


# %% [markdown]
# ## `train`

# %%
# loop one
def train_one(Model, window_i, model_hparams, train_hparams):
    global split_df, targets_df
    
    # set window
    model_hparams.update({'window': split_df.iloc[window_i].window})
    
    # init model
    model = Model(Namespace(**model_hparams))

    # get model type
    train_hparams['task_type'] = model.task_type
    train_hparams['feature_type'] = model.feature_type
    train_hparams['model_type'] = model.model_type
    train_hparams['attn_type'] = model.attn_type

    # checkpoint
    ckpt_prefix = f"{train_hparams['model_type']}_{model_hparams['window']}_"
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        verbose=False,
        mode='min',
        monitor='val_loss',
        prefix=ckpt_prefix,
        filepath=train_hparams['checkpoint_path'],
        save_top_k=train_hparams['save_top_k'],
        period=train_hparams['checkpoint_period'])

    # logger
    logger = pl.loggers.CometLogger(
        api_key='tOoHzzV1S039683RxEr2Hl9PX',
        save_dir='/data/logs',
        project_name='earnings-call',
        experiment_name=model_hparams['window'],
        workspace='amiao',
        display_summary_level=0)

    # early stop
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0.1,
        patience=train_hparams['early_stop_patience'],
        verbose=False,
        mode='min')

    # trainer
    trainer = pl.Trainer(gpus=[0, 1],
                         default_root_dir=train_hparams['checkpoint_path'], 
                         checkpoint_callback=checkpoint_callback, 
                         early_stop_callback=early_stop_callback,
                         overfit_pct=train_hparams['overfit_pct'], 
                         row_log_interval=train_hparams['row_log_interval'],
                         val_check_interval=train_hparams['val_check_interval'], 
                         progress_bar_refresh_rate=2, 
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

    # refresh GPU memory
    refresh_cuda_memory()

    # fit and test
    try:
        # train the model
        trainer.fit(model)

        # load back the best model 
        best_model_name = sorted([f"{train_hparams['checkpoint_path']}/{model_name}" 
                                  for model_name in os.listdir(train_hparams['checkpoint_path']) 
                                  if model_name.startswith(ckpt_prefix)])[-1]
        print(f'loading best model: {best_model_name}')
        best_model = Model.load_from_checkpoint(best_model_name)
        best_model.freeze()

        # test on the best model
        trainer.test(best_model, test_dataloaders=model.test_dataloader())

    except RuntimeError as e:
        raise e
    finally:
        del model, trainer
        refresh_cuda_memory()
        logger.finalize('finished')


# %% [markdown]
# ## `Dataset`

# %%
# Dataset: Txt + Fin-ratio
class CCDataset(Dataset):
    
    def __init__(self, split_window, split_type, text_in_dataset, roll_type, print_window, valid_transcriptids=None, transform=None):
        '''
        Args:
            preembeddings (from globals): list of embeddings. Each element is a tensor (S, E) where S is number of sentences in a call
            targets_df (from globals): DataFrame of targets variables.
            split_df (from globals):
            split_window: str. e.g., "roll-09"
            split_type: str. 'train' or 'test'
            text_only: only output CAR and transcripts if true, otherwise also output financial ratios
            transcriptids: list. If provided, only the given transcripts will be used in generating the Dataset. `transcriptids` is applied **on top of** `split_window` and `split_type`
        '''

        self.text_in_dataset = text_in_dataset
        
        # decalre data as globals so don't   need to create/reload
        global preembeddings, targets_df, split_df
        
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
            
        if split_type=='val':
            transcriptids = targets_df[targets_df.ciq_call_date.between(train_start, train_end)].transcriptid.sample(frac=1, random_state=42).tolist()
            transcriptids = transcriptids[int(len(transcriptids)*0.9):]

        elif split_type=='test':
            transcriptids = targets_df[targets_df.ciq_call_date.between(test_start, test_end)].transcriptid.tolist()

        self.valid_preemb_keys = set(transcriptids).intersection(set(preembeddings.keys()))
        
        if valid_transcriptids is not None:
            self.valid_preemb_keys = self.valid_preemb_keys.intersection(set(valid_transcriptids))
        
        # self attributes
        self.targets_df = targets_df
        self.preembeddings = preembeddings
        self.transform = transform
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
        
        # inputs: preembeddings
        embeddings = self.preembeddings[transcriptid]
        
        # all of the following targests are
        # of type `numpy.float64`
        docid = targets.docid
        
        sue = targets.sue
        sest = targets.sest
        car_0_30 = targets.car_0_30
        
        alpha = targets.alpha
        volatility = targets.volatility
        mcap = targets.mcap/1e6
        bm = targets.bm
        roa = targets.roa
        debt_asset = targets.debt_asset
        numest = targets.numest
        smedest = targets.smedest
        sstdest = targets.sstdest
        car_m1_m1 = targets.car_m1_m1
        car_m2_m2 = targets.car_m2_m2
        car_m30_m3 = targets.car_m30_m3
        volume = targets.volume
        revision = targets.revision
        inflow = targets.inflow/1e3
        
        if self.text_in_dataset:
            return car_0_30, transcriptid, embeddings, alpha, car_m1_m1, car_m2_m2, car_m30_m3, \
                   sest, sue, numest, sstdest, smedest, \
                   mcap, roa, bm, debt_asset, volatility, volume, inflow, revision
        else:
            return docid, \
                   torch.tensor(car_0_30,dtype=torch.float32), \
                   torch.tensor([alpha, car_m1_m1, car_m2_m2, car_m30_m3, sest, sue, numest, sstdest, smedest, mcap, roa, bm, debt_asset, volatility, volume], dtype=torch.float32)


# %% [markdown]
# ## `Model`

# %%
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
        
        self.hparams = hparams
        # self.text_in_dataset will be filled during instanciating.

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
        return {'val_loss': mse, 'log': {'val_rmse': rmse}}
    
    # test step
    def test_epoch_end(self, outputs):
        mse = torch.stack([x['test_loss'] for x in outputs]).mean()
        rmse = torch.sqrt(mse)

        return {'test_loss': mse, 'log': {'test_rmse': rmse}, 'progress_bar':{'test_rmse': rmse}}
    
    # Dataset
    def prepare_data(self):
        self.train_dataset = CCDataset(self.hparams.window, split_type='train', text_in_dataset=self.text_in_dataset,
                                       roll_type=self.hparams.roll_type, print_window=True)
        self.val_dataset = CCDataset(self.hparams.window, split_type='val', text_in_dataset=self.text_in_dataset,
                                     roll_type=self.hparams.roll_type, print_window=False)
        self.test_dataset = CCDataset(self.hparams.window, split_type='test', text_in_dataset=self.text_in_dataset, 
                                      roll_type=self.hparams.roll_type, print_window=False)

    # DataLoader
    def train_dataloader(self):
        # Caution:
        # - If you enable `BatchNorm`, then must set `drop_last=True`.

        collate_fn = self.collate_fn if self.text_in_dataset else None
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True, drop_last=True, num_workers=0, pin_memory=True, collate_fn=collate_fn)
    
    def val_dataloader(self):
        # Caution: 
        # - To improve the validation speed, I'll set val_batch_size to 4. 
        # - Must set `drop_last=True`, otherwise the `val_loss` tensors for different batches won't match and hence give you error.
        # - Not to set `val_batch_size` too large (e.g., 16), otherwise you'll lose precious validation data points
        
        collate_fn = self.collate_fn if self.text_in_dataset else None
        return DataLoader(self.val_dataset, batch_size=self.hparams.val_batch_size, num_workers=0, pin_memory=True, collate_fn=collate_fn, drop_last=True)

    def test_dataloader(self):
        collate_fn = self.collate_fn if self.text_in_dataset else None
        return DataLoader(self.test_dataset, num_workers=0, pin_memory=True, collate_fn=collate_fn)
    
    def collate_fn(self, data):
        '''create mini-batch

        Retures:
            embeddings: tensor, (N, S, E)
            mask: tensor, (N, S)
            sue,car,selead,sest: tensor, (N,)
        '''
        # embeddings: (N, S, E)
        car_0_30, transcriptid, embeddings, alpha, car_m1_m1, car_m2_m2, car_m30_m3, \
        sest, sue, numest, sstdest, smedest, \
        mcap, roa, bm, debt_asset, volatility, volume, inflow, revision = zip(*data)
            
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

        return torch.tensor(car_0_30, dtype=torch.float32), torch.tensor(transcriptid, dtype=torch.float32), \
               embeddings.float(), mask, \
               torch.tensor(alpha, dtype=torch.float32), torch.tensor(car_m1_m1, dtype=torch.float32), \
               torch.tensor(car_m2_m2, dtype=torch.float32), torch.tensor(car_m30_m3, dtype=torch.float32), \
               torch.tensor(sest, dtype=torch.float32), torch.tensor(sue, dtype=torch.float32), \
               torch.tensor(numest, dtype=torch.float32), torch.tensor(sstdest, dtype=torch.float32), \
               torch.tensor(smedest, dtype=torch.float32), torch.tensor(mcap, dtype=torch.float32), \
               torch.tensor(roa, dtype=torch.float32), torch.tensor(bm, dtype=torch.float32), \
               torch.tensor(debt_asset, dtype=torch.float32), torch.tensor(volatility, dtype=torch.float32), \
               torch.tensor(volume, dtype=torch.float32), torch.tensor(inflow, dtype=torch.float32), \
               torch.tensor(revision, dtype=torch.float32)
        
    # optimizer
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer   


# %% [markdown]
# # Transformer

# %%
# MTL-text-fr-Inf
class CCTransformerMTLInf(CC):
    def __init__(self, hparams):
        super().__init__(hparams)
        
        self.hparams = hparams
        
        # specify model type
        self.task_type = 'mtl'
        self.feature_type = 'text + fin-ratio + (inf)'
        self.attn_type = 'dotprod'
        self.model_type = 'transformer'
        self.text_in_dataset = True if self.feature_type!='fin-ratio' else False 

        self.n_covariate = 15
        
        # positional encoding
        self.encoder_pos = PositionalEncoding(hparams.d_model, hparams.attn_dropout)
        
        # encoder layers for input, expert, nonexpert
        encoder_layers_expert = nn.TransformerEncoderLayer(hparams.d_model, hparams.n_head_encoder, hparams.dff, hparams.attn_dropout)
        
        # atten layers
        self.attn_layers_car = nn.Linear(hparams.d_model, 1)
        self.attn_dropout_1 = nn.Dropout(hparams.attn_dropout)
        
        # Build Encoder
        self.encoder_expert = nn.TransformerEncoder(encoder_layers_expert, hparams.n_layers_encoder)
        
        # linear layer to produce final result
        self.linear_car_1 = nn.Linear(hparams.d_model, hparams.d_model)
        self.linear_car_2 = nn.Linear(hparams.d_model, hparams.final_tdim)
        self.linear_car_3 = nn.Linear(hparams.final_tdim+self.n_covariate, hparams.final_tdim+self.n_covariate)
        self.linear_car_4 = nn.Linear(hparams.final_tdim+self.n_covariate, hparams.final_tdim+self.n_covariate)
        self.linear_car_5 = nn.Linear(hparams.final_tdim+self.n_covariate, 1)
        
        self.linear_inflow = nn.Linear(hparams.final_tdim, 1)
        # self.linear_revision = nn.Linear(hparam.final_tdim, 1)
        
        # dropout for final fc layers
        self.final_dropout_1 = nn.Dropout(hparams.dropout)
        self.final_dropout_2 = nn.Dropout(hparams.dropout)
        self.final_dropout_3 = nn.Dropout(hparams.dropout)
        
        # layer normalization
        if hparams.normalize_layer:
            self.layer_norm = nn.LayerNorm(hparams.final_tdim+self.n_covariate)
            
        # batch normalization
        if hparams.normalize_batch:
            self.batch_norm = nn.BatchNorm1d(self.n_covariate)

    # forward
    def forward(self, inp, src_key_padding_mask, alpha, car_m1_m1, car_m2_m2, car_m30_m3, sest, sue, numest, sstdest, \
                smedest, mcap, roa, bm, debt_asset, volatility, volume):
        
        bsz, embed_dim = inp.size(0), inp.size(2)
        
        # if S is longer than max_seq_len, cut
        inp = inp[:,:self.hparams.max_seq_len,] # (N, S, E)
        src_key_padding_mask = src_key_padding_mask[:,:self.hparams.max_seq_len] # (N, S)
        
        inp = inp.transpose(0, 1) # (S, N, E)
        
        # positional encoding
        x = self.encoder_pos(inp) # (S, N, E)
        
        # encode
        x_expert = self.encoder_expert(x, src_key_padding_mask=src_key_padding_mask).transpose(0,1) # (N, S, E)
        
        # multiply with attn
        x_attn = self.attn_dropout_1(F.softmax(self.attn_layers_car(x_expert), dim=1)) # (N, S, 1)
        x_expert = torch.bmm(x_expert.transpose(-1,-2), x_attn).squeeze(-1) # (N, E)
        
        # mix with covariate
        x_expert = self.final_dropout_1(F.relu(self.linear_car_1(x_expert))) # (N, E)
        x_expert = F.relu(self.linear_car_2(x_expert)) # (N, final_tdim)
        
        fin_ratio = torch.cat([alpha.unsqueeze(-1), car_m1_m1.unsqueeze(-1), car_m2_m2.unsqueeze(-1),
                               car_m30_m3.unsqueeze(-1), sest.unsqueeze(-1), sue.unsqueeze(-1), numest.unsqueeze(-1),
                               sstdest.unsqueeze(-1), smedest.unsqueeze(-1), mcap.unsqueeze(-1), roa.unsqueeze(-1), bm.unsqueeze(-1),
                               debt_asset.unsqueeze(-1), volatility.unsqueeze(-1), volume.unsqueeze(-1)], dim=-1) # (N, X)
        
        # batch normalization
        if self.hparams.normalize_batch:
            fin_ratio = self.batch_norm(fin_ratio)
        
        x_car = torch.cat([x_expert, fin_ratio], dim=-1) # (N, X + final_tdim) where X is the number of covariate (n_covariate)

            
        # final FC
        y_inflow = self.linear_inflow(x_expert)
        # y_revision = self.linear_revision(x_expert)
        
        x_car = self.final_dropout_2(F.relu(self.linear_car_3(x_car))) # (N, X + final_tdim)
        y_car = self.linear_car_5(x_car) # (N,1)
        
        # final output
        return y_car, y_inflow
    
    # traning step
    def training_step(self, batch, idx):
        
        car, transcriptid, embeddings, mask, alpha, car_m1_m1, car_m2_m2, car_m30_m3,\
        sest, sue, numest, sstdest, smedest, \
        mcap, roa, bm, debt_asset, volatility, volume, inflow, revision = batch
        
        # get batch size
        bsz = sue.size(0)
        
        # forward
        y_car, y_inflow = self.forward(embeddings, mask, alpha, car_m1_m1, car_m2_m2, car_m30_m3, sest, sue, numest, sstdest, \
                             smedest, mcap, roa, bm, debt_asset, volatility, volume) # (N, 1)
        
        # compute loss
        loss_car = self.mse_loss(y_car, car.unsqueeze(-1)).unsqueeze(-1) # (1,)
        loss_inflow = self.mse_loss(y_inflow, inflow.unsqueeze(-1)).unsqueeze(-1) # (1,)
        
        loss = loss_car + loss_inflow
        
        # logging
        return {'loss': loss, 'log': {'train_loss': loss}}
        
    # validation step
    def validation_step(self, batch, idx):
        car, transcriptid, embeddings, mask, alpha, car_m1_m1, car_m2_m2, car_m30_m3,\
        sest, sue, numest, sstdest, smedest, \
        mcap, roa, bm, debt_asset, volatility, volume, inflow, revision = batch
        
        # get batch size
        bsz = sue.size(0)

        # forward
        y_car, y_inflow = self.forward(embeddings, mask, alpha, car_m1_m1, car_m2_m2, car_m30_m3, sest, sue, numest, sstdest, \
                             smedest, mcap, roa, bm, debt_asset, volatility, volume) # (N, 1)

        # compute loss
        loss_car = self.mse_loss(y_car, car.unsqueeze(-1)).unsqueeze(-1) # (1,)
        loss_inflow = self.mse_loss(y_inflow, inflow.unsqueeze(-1)).unsqueeze(-1) # (1,)
        loss = loss_car + loss_inflow
        
        # logging
        return {'val_car_loss': loss_car, 'val_inflow_loss': loss_inflow, 'val_loss': loss}

    # test step
    def test_step(self, batch, idx):
        car, transcriptid, embeddings, mask, alpha, car_m1_m1, car_m2_m2, car_m30_m3,\
        sest, sue, numest, sstdest, smedest, \
        mcap, roa, bm, debt_asset, volatility, volume, inflow, revision = batch
        
        # get batch size
        bsz = sue.size(0)

        # forward
        y_car, y_inflow = self.forward(embeddings, mask, alpha, car_m1_m1, car_m2_m2, car_m30_m3, sest, sue, numest, sstdest, smedest,\
                             mcap, roa, bm, debt_asset, volatility, volume) # (N, 1)

        # compute loss
        loss_car = self.mse_loss(y_car, car.unsqueeze(-1)).unsqueeze(-1) # (1,)
        loss_inflow = self.mse_loss(y_inflow, inflow.unsqueeze(-1)).unsqueeze(-1) # (1,)
        loss = loss_car + loss_inflow

        # logging
        return {'test_car_loss': loss_car, 'test_inflow_loss': loss_inflow, 'test_loss': loss}  
    
    # epoch_end
    def validation_epoch_end(self, outputs):
        mse = torch.stack([x['val_loss'] for x in outputs]).mean()
        mse_car = torch.stack([x['val_car_loss'] for x in outputs]).mean()
        mse_inflow = torch.stack([x['val_inflow_loss'] for x in outputs]).mean()
        
        rmse = torch.sqrt(mse)
        rmse_car = torch.sqrt(mse_car)
        rmse_inflow = torch.sqrt(mse_inflow)
        
        return {'val_loss': mse, 'log': {'val_rmse': rmse, 'val_car_rmse': rmse_car, 'val_inflow_rmse': rmse_inflow}}
    
    # test step
    def test_epoch_end(self, outputs):
        mse = torch.stack([x['test_loss'] for x in outputs]).mean()
        mse_car = torch.stack([x['test_car_loss'] for x in outputs]).mean()
        mse_inflow = torch.stack([x['test_inflow_loss'] for x in outputs]).mean()
        
        rmse = torch.sqrt(mse)
        rmse_car = torch.sqrt(mse_car)
        rmse_inflow = torch.sqrt(mse_inflow)
        

        return {'test_loss': mse, 'log': {'test_rmse': rmse, 'test_car_rmse': rmse_car, 'test_inflow_rmse': rmse_inflow}}        

# %% [markdown]
# ## run

# %%
# choose Model
Model = CCTransformerMTLInf

# hparams
model_hparams = {
    'preembedding_type': 'all_sbert_roberta_nlistsb_encoded', # key!
    'targets_name': 'f_sue_keydevid_car_finratio_vol_transcriptid_sim_inflow_revision_text', # key!
    'roll_type': '3y',  # key!
    'batch_size': 32,
    'val_batch_size': 4,
    'max_seq_len': 768, 
    'learning_rate': 3e-4,
    'task_weight': 1,
    'normalize_layer': False, # key!
    'normalize_batch': True, # key!

    'n_layers_encoder': 6,
    'n_head_encoder': 8, 
    'd_model': 1024,
    'final_tdim': 1024, 
    'dff': 2048,
    'attn_dropout': 0.1,
    'dropout': 0.5,
    'n_head_decoder': 8} 

train_hparams = {
    # log
    'note': 'temp',
    'checkpoint_path': CHECKPOINT_DIR,
    'row_log_interval': 1,
    'save_top_k': 1,
    'val_check_interval': 0.2,

    # data size
    'overfit_pct': 1,
    'min_epochs': 3,
    'max_epochs': 25,
    'max_steps': None,
    'accumulate_grad_batches': 1,

    # Caution:
    # The check of patience depends on **how often you compute your val_loss** (`val_check_interval`). 
    # Say you check val every N baches, then `early_stop_callback` will compare to your latest N **baches**.
    # If you compute val_loss every N **epoches**, then `early_stop_callback` will compare to the latest N **epochs**.
    'early_stop_patience': 10,

    # Caution:
    # If set to 1, then save ckpt every 1 epoch
    # If set to 0, then save ckpt on every val!!! (if val improves)
    'checkpoint_period': 0}

# delete all existing .ckpt files
refresh_ckpt(train_hparams['checkpoint_path'])

# load split_df
load_split_df(model_hparams['roll_type'])
    
# load targets_df
load_targets(model_hparams['targets_name'])

# loop over 24!
for window_i in range(len(split_df)):
    # load preembeddings
    load_preembeddings(model_hparams['preembedding_type'])

    # train one window
    train_one(Model, window_i, model_hparams, train_hparams)

# %%
