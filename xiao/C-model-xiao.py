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
    ckpt_prefix = f"{train_hparams['note']}_{model_hparams['window']}_"
    
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        verbose=True,
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
    trainer = pl.Trainer(gpus=-1, 
                         precision=train_hparams['precision'],
                         checkpoint_callback=checkpoint_callback, 
                         early_stop_callback=early_stop_callback,
                         overfit_batches=train_hparams['overfit_batches'], 
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


# %% [markdown]
# ## `Dataset`

# %%
# Dataset: Txt + Fin-ratio
class CCDataset(Dataset):
    
    def __init__(self, split_window, split_type, text_in_dataset, roll_type, print_window, preembeddings, targets_df, split_df, valid_transcriptids=None):
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
                   car_0_30, \
                   car_0_30_norm, \
                   [alpha, car_m1_m1, car_m2_m2, car_m30_m3, sest, sue, numest, sstdest, smedest, mcap, roa, bm, debt_asset, volatility, volume]


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
        
        self.hparams = Namespace(**hparams)
        # self.text_in_dataset will be filled during instanciating.
        
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
        return {'val_loss': mse, 'log': {'val_rmse': rmse}}
    
    # test step
    def test_epoch_end(self, outputs):
        mse = torch.stack([x['test_loss'] for x in outputs]).mean()
        rmse = torch.sqrt(mse)

        return {'test_loss': mse, 'log': {'test_rmse': rmse}, 'progress_bar':{'test_rmse': rmse}}
    
    # Dataset
    def prepare_data(self):
        
        self.train_dataset = CCDataset(self.hparams.window, split_type='train', text_in_dataset=self.text_in_dataset,
                                       roll_type=self.hparams.roll_type, print_window=True, preembeddings=self.preembeddings,
                                       targets_df=self.targets_df, split_df=self.split_df)
        self.val_dataset = CCDataset(self.hparams.window, split_type='val', text_in_dataset=self.text_in_dataset,
                                     roll_type=self.hparams.roll_type, print_window=False, preembeddings=self.preembeddings,
                                       targets_df=self.targets_df, split_df=self.split_df)
        self.test_dataset = CCDataset(self.hparams.window, split_type='test', text_in_dataset=self.text_in_dataset, 
                                      roll_type=self.hparams.roll_type, print_window=False, preembeddings=self.preembeddings,
                                       targets_df=self.targets_df, split_df=self.split_df)

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


# %% [markdown]
# # STL

# %% [markdown]
# ## model

# %%
# STL-text-fr
class CCTransformerSTLTxtFr(CC):
    def __init__(self, hparams):
        # `self.hparams` will be created by super().__init__
        super().__init__(hparams)
        
        # specify model type
        self.model_type = 'TSFM'
        self.target_type = 'car'
        self.feature_type = 'txt'
        self.normalize_target = True
        self.attn_type = 'dotprod'
        self.text_in_dataset = True if self.feature_type!='fr' else False 

        self.n_covariate = 15
        
        # positional encoding
        self.encoder_pos = PositionalEncoding(self.hparams.d_model, self.hparams.attn_dropout)
        
        # encoder layers for input, expert, nonexpert
        encoder_layers_expert = nn.TransformerEncoderLayer(self.hparams.d_model, self.hparams.n_head_encoder, self.hparams.dff, self.hparams.attn_dropout)
        
        # atten layers for SUE, CAR, SELEAD, SEST
        self.attn_layers_car = nn.Linear(self.hparams.d_model, 1)
        self.attn_dropout_1 = nn.Dropout(self.hparams.attn_dropout)
        
        # Build Encoder and Decoder
        self.encoder_expert = nn.TransformerEncoder(encoder_layers_expert, self.hparams.n_layers_encoder)
        
        # linear layer to produce final result
        self.txt_fc_1 = nn.Linear(self.hparams.d_model, self.hparams.d_model)
        self.fc_1 = nn.Linear(self.hparams.d_model+self.n_covariate, self.hparams.d_model+self.n_covariate)
        self.fc_2 = nn.Linear(self.hparams.d_model+self.n_covariate, self.hparams.d_model+self.n_covariate)
        self.fc_3 = nn.Linear(self.hparams.d_model+self.n_covariate, 1)
        
        # dropout for final fc layers
        self.txt_dropout_1 = nn.Dropout(self.hparams.dropout)
        
    # forward
    def forward(self, embeddings, src_key_padding_mask, fin_ratios):
        
        # if S is longer than max_seq_len, cut
        embeddings = embeddings[:,:self.hparams.max_seq_len,] # (N, S, E)
        src_key_padding_mask = src_key_padding_mask[:,:self.hparams.max_seq_len] # (N, S)
        
        embeddings = embeddings.transpose(0, 1) # (S, N, E)
        
        # positional encoding
        x = self.encoder_pos(embeddings) # (S, N, E)
        
        # encode
        x_expert = self.encoder_expert(x, src_key_padding_mask=src_key_padding_mask).transpose(0,1) # (N, S, E)
        
        # decode with attn
        x_attn = self.attn_dropout_1(F.softmax(self.attn_layers_car(x_expert), dim=1)) # (N, S, 1)
        x_expert = torch.bmm(x_expert.transpose(-1,-2), x_attn).squeeze(-1) # (N, E)
        
        x_expert = self.txt_dropout_1(self.txt_fc_1(x_expert)) 
        
        
        # Since we use the model as feature extractor, we won't include `fin_ratios`
        # concate `x_final` with `fin_ratios`
        x_final = torch.cat([x_expert, fin_ratios], dim=-1) # (N, E+X) where X is the number of covariate (n_covariate)
        
        # final FC
        x_final = F.relu(self.fc_1(x_final)) # (N, E+X)
        x_final = F.relu(self.fc_2(x_final)) # (N, E+X)
        y_car = self.fc_3(x_final) # (N, 1)
        
        # final output
        return y_car
    
    # traning step
    def training_step(self, batch, idx):
        car, car_norm, inflow, inflow_norm, revision, revision_norm, \
        transcriptid, embeddings, mask, \
        fin_ratios = batch
        
        # forward
        y_car = self.forward(embeddings, mask, fin_ratios) # (N, 1)

        # compute loss
        loss_car = self.mse_loss(y_car, car_norm.unsqueeze(-1)).unsqueeze(-1) # (1,)
        
        # logging
        return {'loss': loss_car, 'log': {'train_loss': loss_car}}
        
    # validation step
    def validation_step(self, batch, idx):
        car, car_norm, inflow, inflow_norm, revision, revision_norm, \
        transcriptid, embeddings, mask, \
        fin_ratios = batch
        
        # forward
        y_car = self.forward(embeddings, mask, fin_ratios) # (N, 1)

        # compute loss
        loss_car = self.mse_loss(y_car, car_norm.unsqueeze(-1)).unsqueeze(-1) # (1,)

        # logging
        return {'val_loss': loss_car}

    # test step
    def test_step(self, batch, idx):
        car, car_norm, inflow, inflow_norm, revision, revision_norm, \
        transcriptid, embeddings, mask, \
        fin_ratios = batch
        
        # forward
        y_car = self.forward(embeddings, mask, fin_ratios) # (N, 1)

        # compute loss
        loss_car = self.mse_loss(y_car, car_norm.unsqueeze(-1)).unsqueeze(-1) # (1,)

        # logging
        return {'test_loss': loss_car}  

# %% [markdown]
# ## run

# %%
# choose Model
Model = CCTransformerSTLTxtFr

# hparams
model_hparams = {
    'preembedding_type': 'all_sbert_roberta_nlistsb_encoded', # key!
    'targets_name': 'f_sue_keydevid_car_finratio_vol_transcriptid_sim_inflow_revision_text_norm', # key!
    'roll_type': '3y',  # key!
    
    # task weight
    'car_weight': 1, # key!
    'inflow_weight': 0, # key!
    
    'batch_size': 28,
    'val_batch_size': 4,
    'max_seq_len': 768, 
    'learning_rate':1e-4, 
    'task_weight': 1,
    'n_layers_encoder': 4,
    'n_head_encoder': 8, 
    'd_model': 1024,
    'final_tdim': 1024, 
    'dff': 2048,
    'attn_dropout': 0.1,
    'dropout': 0.5,
    'n_head_decoder': 8} 

train_hparams = {
    # log
    'machine': 'ASU',  # key!
    'note': f"STL-10,(car~txt+fr),txtfc=1,fc=2,bsz={model_hparams['batch_size']},lr={model_hparams['learning_rate']}", # key!
    'row_log_interval': 10,
    'save_top_k': 1,
    'val_check_interval': 0.2,

    # data size
    'precision': 32,
    'overfit_batches': 0.0,
    'min_epochs': 3,
    'max_epochs': 20,
    'max_steps': None,
    'accumulate_grad_batches': 1,

    # Caution:
    # The check of patience depends on **how often you compute your val_loss** (`val_check_interval`). 
    # Say you check val every N baches, then `early_stop_callback` will compare to your latest N **baches**.
    # If you compute val_loss every N **epoches**, then `early_stop_callback` will compare to the latest N **epochs**.
    'early_stop_patience': 8,

    # Caution:
    # If set to 1, then save ckpt every 1 epoch
    # If set to 0, then save ckpt on every val!!! (if val improves)
    'checkpoint_period': 0}

# delete all existing .ckpt files
refresh_ckpt()

# load split_df
load_split_df(model_hparams['roll_type'])
    
# load targets_df
load_targets(model_hparams['targets_name'])

# load preembeddings
load_preembeddings(model_hparams['preembedding_type'])
    
# loop over 24!
for window_i in range(len(split_df)):

    # train one window
    train_one(Model, window_i, model_hparams, train_hparams)
