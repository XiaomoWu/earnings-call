# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.9.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Init

# +
# import tensorflow as tf
import comet_ml
import datatable as dt
import gc
import numpy as np
import torch
import os
import pytorch_lightning as pl
import torch.nn.functional as F
import torch.optim as optim
import shutil
import pandas as pd
import pyarrow.feather as feather

from argparse import Namespace
from collections import OrderedDict, defaultdict
from datatable import f, update
from datetime import datetime
from itertools import chain
from operator import itemgetter
from pytorch_lightning.loggers import CometLogger
from tqdm.auto import tqdm
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data.dataset import random_split
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset, DataLoader

# Init for script use
with open("/home/yu/OneDrive/App/Settings/jupyter + R + Python/python_startup.py", 'r') as _:
    exec(_.read())

os.chdir('/home/yu/OneDrive/CC')

# working directory
ROOT_DIR = '/home/yu/OneDrive/CC'
DATA_DIR = f'{ROOT_DIR}/data'
CHECKPOINT_DIR = '/home/yu/Data/CC-checkpoints'
CHECKPOINT_TEMP_DIR = f'{CHECKPOINT_DIR}/temp'

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


# -

# # Base

# ## helpers

# +
# helper: tensor_to_list
def tensor_to_str(tensor, tailing='\n'):
    '''Given a 1d tensor, convert it to a list of string
    
    tailing: str. Added to the end of every element
    '''
    assert isinstance(tensor, torch.Tensor), 'Give me a tensor please!'
    assert len(tensor.shape)==1, 'Must be 1D tensor!'
    tensor = tensor.to('cpu').tolist()
    tensor = [f'{str(x)}{tailing}' for x in tensor]
    return tensor


# helper: refresh cuda memory
def refresh_cuda_memory():
    """
    Re-allocate all cuda memory to help alleviate fragmentation
    """
    # Run a full garbage collect first so any dangling tensors are released
    gc.collect()

    # Then move all tensors to the CPU
    for obj in gc.get_objects():
        if isinstance(obj, torch.Tensor) and obj.device!=torch.device('cpu'):
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
def load_targets(targets_name, force=False):
    targets_df = feather.read_feather(f'{DATA_DIR}/{targets_name}.feather')
    targest_df = targets_df[targets_df.outlier_flag1==False]
    return targets_df
        
# helpers: load preembeddings
def load_preembeddings(preembedding_name):
    if 'preembeddings' not in globals():
        
        # find the embedding files
        emb_paths = [path for path in os.listdir('data/Embeddings')
                     if re.search(f'{preembedding_name}_rank', path)]
        emb_paths.sort()
        assert len(emb_paths)==2, "Expect two files: rank0 and rank1"

        # load the embedding files
        print(f'{datetime.now().strftime("%Y-%m-%d %I:%M:%S %p")}, Loading "{emb_paths[0]}"...')
        emb0 = torch.load(f"{DATA_DIR}/Embeddings/{emb_paths[0]}")
        print(f'{datetime.now().strftime("%Y-%m-%d %I:%M:%S %p")}, Loading "{emb_paths[1]}"...')
        emb1 = torch.load(f"{DATA_DIR}/Embeddings/{emb_paths[1]}")

        # merge two ranks into one (update emb0 with emb1)
        for tid, cid_emb in emb1.items():
            for cid, emb in cid_emb.items():
                emb0[tid].update({cid:emb})
        print('Merging completes!')

        # write embedding to globals()
        globals()['preembeddings'] = emb0
    
    else:
        print(f'Pre-embedding "{preembedding_name}" already loaded, will not load again!')

# helpers: load split_df
def load_split_df(window_size):
    split_df = pd.read_csv(f'{DATA_DIR}/split_dates.csv')
    return split_df.loc[split_df.window_size==window_size]

# helpers: load tid_cid_pair
def load_tid_cid_pair(tid_cid_pair_name):
    '''load DataFrame tid_cid_pair, convert it into a Dict
    
    output: {tid:[cid1, cid2, ...]}
    
    tid_cid_pair_name: str. e.g., "md", "qa", "all"
    '''
    pair = feather.read_feather(f'data/tid_cid_pair_{tid_cid_pair_name}.feather')
    output = {}
    for tid, group in pair.groupby(['transcriptid']):
        cids = group.componentid.to_list()
        output[tid] = cids
    return output
    
# helpers: load tid_cid_pair
def load_tid_from_to_pair(tid_from_to_pair_name):
    '''load DataFrame tid_from_to_pair, convert it into a Dict
    
    output: {tid_from:[tid_to1, tid_to2, ...]}
    
    tid_cid_pair_name: str. e.g., "3qtr"
    '''
    pair = feather.read_feather(f'data/tid_from_to_pair_{tid_from_to_pair_name}.feather')
    
    output = {}
    for _, (_, tid_from, tid_to) in pair.iterrows():
        output[tid_from] = tid_to.tolist()
    return output
    
# helper: log_ols_rmse
def log_ols_rmse(logger, yqtr, window_size):
    '''
    Given yqtr, find the corresponding ols_rmse from `performance_by_model.feather`.
    Always compare to the same model: 'ols: car_norm ~ fr'
    then log to Comet
    '''
    performance = dt.Frame(pd.read_feather('data/performance_by_yqtr.feather'))


    ols_rmse_norm = performance[(f.model_name=='ols: car_norm ~ fr') & (f.window_size==window_size) & (f.yqtr==yqtr), f.rmse][0,0]
    logger.experiment.log_parameter('ols_rmse_norm', ols_rmse_norm)
    
def log_test_start(logger, window_size, yqtr):
    '''
    Given window, find the corresponding star/end date of the training/test periods, 
    then log to Comet
    '''
    split_df = pd.read_csv(f'data/split_dates.csv')

    _, train_start, train_end, test_start, test_end, *_ = tuple(split_df.loc[(split_df.yqtr==yqtr) & (split_df.window_size==window_size)].iloc[0])
    
    logger.experiment.log_parameter('train_start', train_start)
    logger.experiment.log_parameter('train_end', train_end)
    logger.experiment.log_parameter('test_start', test_start)
    logger.experiment.log_parameter('test_end', test_end)


# -

# ## def Data

# +
# Define Dataset
class CCDataset(Dataset):
    
    def __init__(self, yqtr, split_type, text_in_dataset, window_size, preembeddings, targets_df, split_df, tid_cid_pair, tid_from_to_pair):
        '''
        Args:
            preembeddings: dict of pre-embeddings. In the form
              `{tid:{cid:{'embedding':Tensor, other-key-value-pair}}}` 
              for component level and 
              `{tid:{sid:{'embedding':Tensor, other-key-value-pair}}}` 
              for sentence level.
              
            targets_df: DataFrame of targets variables.
            split_df: DataFrame that keeps the split of windows
            ytqr: str. e.g., "2008-q3"
            split_type: str. 'train', 'val', or 'test'
            text_in_dataset: also output text embedding if true.
            
            tid_cid_pair: Dict of transcriptid and componentid/sentenceid for
              text that will be used. In the form 
              `{tid:[cid1, cid2, ...]}` or `{tid:[sid1, sid2, ...]}`
              Note! [cid1, cid2, ...] must be in the same order as in original 
              transcript!
        '''
            
        # get split dates from `split_df`
        _, train_start, train_end, test_start, test_end, _, yqtr = tuple(split_df.loc[(split_df.yqtr==yqtr) & (split_df.window_size==window_size)].iloc[0])
        
        train_start = datetime.strptime(train_start, '%Y-%m-%d').date()
        train_end = datetime.strptime(train_end, '%Y-%m-%d').date()
        test_start = datetime.strptime(test_start, '%Y-%m-%d').date()
        test_end = datetime.strptime(test_end, '%Y-%m-%d').date()
        
        # generate targets_df for train, val, test 
        if split_type=='train':
            # print current window
            print(f'Current window: {yqtr} ({window_size}) \n(train: {train_start} to {train_end}) (test: {test_start} to {test_end})')
            
            targets_df = targets_df[targets_df.ciq_call_date.between(train_start, train_end)].sample(frac=1, random_state=42)
            targets_df = targets_df.iloc[:int(len(targets_df)*0.9)]
            
        elif split_type=='val':
            targets_df = targets_df[targets_df.ciq_call_date.between(train_start, train_end)].sample(frac=1, random_state=42)
            targets_df = targets_df.iloc[int(len(targets_df)*0.9):]

        elif split_type=='test':
            targets_df = targets_df[targets_df.ciq_call_date.between(test_start, test_end)]

        
        # make sure targets_df only contains transcriptid that're also 
        # in preembeddings
        targets_df = targets_df.loc[targets_df.transcriptid.isin(set(preembeddings.keys()))]
        
        # Assign states
        self.text_in_dataset = text_in_dataset
        if text_in_dataset:
            self.preembeddings = preembeddings
        self.targets_df = targets_df
        self.train_start = train_start
        self.train_end = train_end
        self.test_start = test_start
        self.test_end = test_end
        self.split_type = split_type
        self.tid_cid_pair = tid_cid_pair
        self.tid_from_to_pair = tid_from_to_pair
        
    def __len__(self):
        return len(self.targets_df)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        targets = self.targets_df.iloc[idx]
        
        # all of the following targests are
        # of type `numpy.float64`
        transcriptid = targets.transcriptid
        car_0_30 = targets.car_0_30
        car_0_30_norm = targets.car_0_30_norm
        revision = targets.revision
        revision_norm = targets.revision_norm
        inflow = targets.inflow
        inflow_norm = targets.inflow_norm
        
        # using the normalized features
        similarity = targets.similarity_bigram_norm
        sentiment = targets.qa_positive_sent_norm
        sue = targets.sue_norm
        sest = targets.sest_norm        
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
            embeddings = assemble_embedding(transcriptid, 
                                            self.preembeddings,
                                            self.tid_cid_pair,
                                            self.tid_from_to_pair)

            return car_0_30, car_0_30_norm, inflow, inflow_norm, revision,\
                   revision_norm,  transcriptid, embeddings, \
                   [alpha, car_m1_m1, car_m2_m2, car_m30_m3, sest, sue, numest, sstdest, smedest, mcap, roa, bm, debt_asset, volatility, volume]
        else:
            return torch.tensor(transcriptid,dtype=torch.int64), \
                   torch.tensor(car_0_30,dtype=torch.float32), \
                   torch.tensor(car_0_30_norm,dtype=torch.float32), \
                   torch.tensor([similarity, sentiment],
                                dtype=torch.float32),\
                   torch.tensor([alpha, car_m1_m1, car_m2_m2, car_m30_m3,\
                                 sest, sue, numest, sstdest, smedest, mcap,\
                                 roa, bm, debt_asset, volatility, volume],
                                dtype=torch.float32)
      
    
def assemble_embedding(transcriptid, preembeddings, 
                       tid_cid_pair, tid_from_to_pair):
    '''Assemble embeddings belonging to the same tid into one Tensor
    
    Method:
        1) Given transcriptid, use it as "transcriptid_from" to retrieve all the 
           corresponding "transcriptid_to" from table "tid_from_to_pair"
        2) For every transcript_to, retrieve all the corresponding cids from table
           "tid_cid_pair"
    '''
    # find tids that we'll consider
    tids_to = tid_from_to_pair[transcriptid]
    
    # for every tid, merge its components
    output = []
    
    for tid_to in tids_to:
        comps = preembeddings[tid_to]
        emb = [comps[cid]['embedding'] 
               for cid in tid_cid_pair.get(tid_to, [])]
        output.extend(emb)
        
    return torch.stack(output)


# -

# then define DataModule
class CCDataModule(pl.LightningDataModule):
    def __init__(self, yqtr, preembedding_name, targets_name,
                 batch_size, val_batch_size, test_batch_size,
                 text_in_dataset, window_size, tid_cid_pair_name,
                 tid_from_to_pair_name):
        super().__init__()
        
        self.yqtr = yqtr
        self.preembedding_name = preembedding_name
        self.targets_name = targets_name
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.text_in_dataset = text_in_dataset
        self.window_size = window_size
        self.tid_cid_pair_name = tid_cid_pair_name
        self.tid_from_to_pair_name = tid_from_to_pair_name
        
    # Dataset
    def setup(self):
        # read the preembedding, targests, and split_df
        global preembeddings
        
        load_preembeddings(self.preembedding_name)
        targets_df = load_targets(self.targets_name)
        split_df = load_split_df(self.window_size)
        tid_cid_pair = load_tid_cid_pair(self.tid_cid_pair_name)
        tid_from_to_pair = load_tid_from_to_pair(self.tid_from_to_pair_name)
        
        self.train_dataset = CCDataset(self.yqtr, 
                                       split_type='train',
                                       text_in_dataset=self.text_in_dataset,
                                       window_size=self.window_size,
                                       preembeddings=preembeddings,
                                       targets_df=targets_df, 
                                       split_df=split_df,
                                       tid_cid_pair=tid_cid_pair,
                                       tid_from_to_pair=tid_from_to_pair)
        print(f'N train = {len(self.train_dataset)}')
        
        self.val_dataset = CCDataset(self.yqtr, split_type='val',
                                     text_in_dataset=self.text_in_dataset,
                                     window_size=self.window_size,
                                     preembeddings=preembeddings,
                                     targets_df=targets_df,
                                     split_df=split_df,
                                     tid_cid_pair=tid_cid_pair,
                                     tid_from_to_pair=tid_from_to_pair)
        print(f'N val = {len(self.val_dataset)}')
        print(f'N train+val = {len(self.train_dataset)+len(self.val_dataset)}')

        self.test_dataset = CCDataset(self.yqtr, split_type='test',
                                      text_in_dataset=self.text_in_dataset, 
                                      window_size=self.window_size,
                                      preembeddings=preembeddings,
                                      targets_df=targets_df,
                                      split_df=split_df,
                                      tid_cid_pair=tid_cid_pair,
                                      tid_from_to_pair=tid_from_to_pair)
        print(f'N test = {len(self.test_dataset)}')

    # DataLoader
    def train_dataloader(self):
        # Caution:
        # - If you enable `BatchNorm`, then must set `drop_last=True`.

        collate_fn = self.collate_fn if self.text_in_dataset else None
        return DataLoader(self.train_dataset, batch_size=self.batch_size, 
                          shuffle=True, drop_last=False, num_workers=2,
                          pin_memory=True, collate_fn=collate_fn)
    
    def val_dataloader(self):
        # Caution: 
        # - To improve the validation speed, I'll set val_batch_size to 4. 
        # - It's safe to set `drop_last=True` without under-counting samples.
        
        collate_fn = self.collate_fn if self.text_in_dataset else None
        return DataLoader(self.val_dataset, batch_size=self.val_batch_size,
                          num_workers=2, pin_memory=True, collate_fn=collate_fn,
                          drop_last=False)

    def test_dataloader(self):
        collate_fn = self.collate_fn if self.text_in_dataset else None
        return DataLoader(self.test_dataset, batch_size=self.test_batch_size, num_workers=2, 
                          pin_memory=True, collate_fn=collate_fn, drop_last=False)
    
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
        
        return torch.tensor(car_0_30, dtype=torch.float32), \
               torch.tensor(car_0_30_norm, dtype=torch.float32), \
               torch.tensor(inflow, dtype=torch.float32), \
               torch.tensor(inflow_norm, dtype=torch.float32), \
               torch.tensor(revision, dtype=torch.float32), \
               torch.tensor(revision_norm, dtype=torch.float32), \
               torch.tensor(transcriptid, dtype=torch.float32), \
               embeddings.float(), mask, \
               torch.tensor(fin_ratios, dtype=torch.float32)


# ## def Model

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
    '''Mainly define the `*_step_end` methods
    '''
    def __init__(self, learning_rate):
        super().__init__()
        self.save_hyperparameters()
        
    # forward
    def forward(self):
        pass
    
    # loss
    def mse_loss(self, y, t):
        return F.mse_loss(y, t)
        
    # def training_step_end
    def training_step_end(self, outputs):
        y = outputs['y_car']
        t = outputs['t_car']
        loss = self.mse_loss(y, t)
        
        return {'loss':loss}
    
    # def validation_step_end
    def validation_step_end(self, outputs):
        y_car = outputs['y_car']
        t_car = outputs['t_car']
        
        return {'y_car':y_car, 't_car':t_car}
        
    # validation step
    def validation_epoch_end(self, outputs):
        '''
        outputs: a list. len(outputs) == num_steps.
            e.g., outputs = [{'val_loss': 3}, {'val_loss': 4}]
        '''
        y_car = torch.cat([x['y_car'] for x in outputs])
        t_car = torch.cat([x['t_car'] for x in outputs])
        
        rmse = torch.sqrt(self.mse_loss(y_car, t_car))
        self.log('val_rmse', rmse, on_step=False)
        
    # test step
    def test_step_end(self, outputs):
        transcriptid = outputs['transcriptid']
        
        y_car = outputs['y_car']
        t_car = outputs['t_car']
        
        return {'y_car':y_car, 't_car':t_car, 'transcriptid':transcriptid}
    
    def test_epoch_end(self, outputs):
        
        transcriptid = torch.cat([x['transcriptid'] for x in outputs])
        y_car = torch.cat([x['y_car'] for x in outputs])
        t_car = torch.cat([x['t_car'] for x in outputs])
        
        rmse = torch.sqrt(self.mse_loss(y_car, t_car))
        self.log('test_rmse', rmse, on_step=False)
        
        if 'test_loss_car' in outputs[0]:
            rmse_car = torch.sqrt(torch.stack([x['test_loss_car'] for x in outputs]).mean())
            self.log('test_rmse_car', rmse_car, on_step=False)
            
        if 'test_loss_inflow' in outputs[0]:
            rmse_inflow = torch.sqrt(torch.stack([x['test_loss_inflow'] for x in outputs]).mean())
            self.log('test_rmse_inflow', rmse_inflow, on_step=False)

        if 'test_loss_revision' in outputs[0]:
            rmse_revision = torch.sqrt(torch.stack([x['test_loss_revision'] for x in outputs]).mean())
            self.log('test_rmse_revision', rmse_revision, on_step=False)
        
        # save & log `y_car`
        y_car_filename = f'{DATA_DIR}/y_car.feather'
        
        df = pd.DataFrame({'transcriptid':transcriptid.to('cpu'), 'y_car':y_car.to('cpu')})
        feather.write_feather(df, y_car_filename)
            
        self.logger.experiment.log_asset(y_car_filename)
            
    # optimizer
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer   


# -

# ## def train()

# loop one
def train_one(Model, yqtr, data_hparams, model_hparams, trainer_hparams):

    # ----------------------
    # `hparams` sanity check
    # ----------------------
    
    # check: batch_size//len(gpus)==0
    assert data_hparams['batch_size']%len(trainer_hparams['gpus'])==0, \
        f"`batch_size` must be divisible by `len(gpus)`. Currently batch_size={model_hparams['batch_size']}, gpus={trainer_hparams['gpus']}"
    
    # check: val_batch_size//len(gpus)==0
    assert data_hparams['val_batch_size']%len(trainer_hparams['gpus'])==0, \
        f"`val_batch_size` must be divisible by `len(gpus)`. Currently batch_size={model_hparams['val_batch_size']}, gpus={trainer_hparams['gpus']}"
    
    # check: val_batch_size//len(gpus)==0
    assert data_hparams['test_batch_size']%len(trainer_hparams['gpus'])==0, \
        f"`test_batch_size` must be divisible by `len(gpus)`. Currently batch_size={model_hparams['test_batch_size']}, gpus={trainer_hparams['gpus']}"
    
    # ----------------------------
    # Initialize model and trainer
    # ----------------------------
    
    # init model
    model = Model(**model_hparams)

    # checkpoint
    ckpt_prefix = f"{trainer_hparams['note']}_{data_hparams['yqtr']}_".replace('*', '')
    
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        verbose=True,
        mode='min',
        monitor='val_rmse',
        filepath=CHECKPOINT_DIR,
        prefix=ckpt_prefix,
        save_top_k=trainer_hparams['save_top_k'],
        period=trainer_hparams['checkpoint_period'])

    # logger
    logger = CometLogger(
        api_key=COMET_API_KEY,
        save_dir='/data/logs',
        project_name='earnings-call',
        experiment_name=data_hparams['yqtr'],
        workspace='amiao',
        display_summary_level=0)

    # early stop
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor='val_rmse',
        min_delta=0,
        patience=trainer_hparams['early_stop_patience'],
        verbose=True,
        mode='min')

    # trainer
    trainer = pl.Trainer(gpus=trainer_hparams['gpus'], 
                         precision=trainer_hparams['precision'],
                         checkpoint_callback=checkpoint_callback, 
                         callbacks=[early_stop_callback],
                         overfit_batches=trainer_hparams['overfit_batches'], 
                         log_every_n_steps=trainer_hparams['log_every_n_steps'],
                         val_check_interval=trainer_hparams['val_check_interval'], 
                         progress_bar_refresh_rate=5, 
                         distributed_backend='ddp', 
                         accumulate_grad_batches=trainer_hparams['accumulate_grad_batches'],
                         min_epochs=trainer_hparams['min_epochs'],
                         max_epochs=trainer_hparams['max_epochs'], 
                         max_steps=trainer_hparams['max_steps'], 
                         logger=logger)

    # add n_model_params
    trainer_hparams['n_model_params'] = sum(p.numel() for p in model.parameters())

    # upload hparams
    logger.experiment.log_parameters(data_hparams)
    logger.experiment.log_parameters(model_hparams)
    logger.experiment.log_parameters(trainer_hparams)
    
    # upload ols_rmse (for reference)
    log_ols_rmse(logger, data_hparams['yqtr'], data_hparams['window_size'])
    
    # upload test_start
    log_test_start(logger, data_hparams['window_size'], data_hparams['yqtr'])
    
    # If run on ASU, upload code explicitly
    if trainer_hparams['machine'] == 'ASU':
        codefile = [name for name in os.listdir('.') if name.endswith('.py')]
        assert len(codefile)==1, f'There must be only one `.py` file in the current directory! {len(codefile)} files detected: {codefile}'
        logger.experiment.log_asset(codefile[0])
    
    
    # refresh GPU memory
    refresh_cuda_memory()

    
    # ----------------------------
    # fit and test
    # ----------------------------

    try:
        # create datamodule
        datamodule = CCDataModule(**data_hparams)
        datamodule.setup()

        # train the model
        trainer.fit(model, datamodule)

        # test on the best model
        trainer.test(ckpt_path=None)

    except RuntimeError as e:
        raise e
    finally:
        del model, trainer
        refresh_cuda_memory()
        logger.finalize('finished')


# + [markdown] toc-hr-collapsed=true
# # MLP
# -

# ## model

# MLP
class CCMLP(CC):
    def __init__(self, learning_rate, dropout, model_type='MLP'):
        super().__init__(learning_rate)
        
        self.save_hyperparameters()
        
        # dropout layers
        # self.dropout_1 = nn.Dropout(self.hparams.dropout)
        # self.dropout_2 = nn.Dropout(self.hparams.dropout)
        
        # fc layers
        self.fc_1 = nn.Linear(17, 32)
        self.fc_2 = nn.Linear(32, 1)
        #self.fc_3 = nn.Linear(32, 1)
        
    def forward(self):
        pass
    
    def shared_step(self, batch):
        transcriptid, car, car_norm, manual_txt, fin_ratios = batch
        x = torch.cat([fin_ratios, manual_txt], dim=-1) # (N, 2+15)

        x_car = F.relu(self.fc_1(x))
        y_car = self.fc_2(x_car) # (N, 1)    
        
        t_car = car_norm
        
        return transcriptid.squeeze(), y_car.squeeze(), t_car.squeeze() 
        
    # train step
    def training_step(self, batch, idx):
        transcriptid, y_car, t_car = self.shared_step(batch)
        return {'y_car': y_car, 't_car': t_car}
        
    # validation step
    def validation_step(self, batch, idx):
        transcriptid, y_car, t_car = self.shared_step(batch)
        return {'y_car': y_car, 't_car': t_car}
        
    # test step
    def test_step(self, batch, idx):
        transcriptid, y_car, t_car = self.shared_step(batch)
        return {'transcriptid':transcriptid, 'y_car':y_car, 't_car': t_car}  

# ## run

'''
# choose Model
Model = CCMLP

# data hparams
data_hparams = {
    'preembedding_name': 'all_sbert_roberta_nlistsb_encoded', # key!
    'targets_name': 'f_sue_keydevid_car_finratio_vol_transcriptid_sim_inflow_revision_retail_sentiment_norm_outlier', # key!

    'batch_size': 128,
    'val_batch_size':64,
    'test_batch_size':32,
    
    'text_in_dataset': False,
    'window_size': '5y', # key!
}

# model hparams
model_hparams = {
    'learning_rate': 1e-3,
    'dropout': 0.5,
}

# train hparams
trainer_hparams = {
    # random seed
    'seed': 42,    # key
    
    # gpus
    'gpus': [0,1], # key

    # checkpoint & log
    
    # last: MLP-24
    'machine': 'yu-workstation', # key!
    'note': f"MLP-25,(car~fr+mtxt),hidden=32,hiddenLayer=1,fc_dropout=no,NormCAR=yes,bsz={data_hparams['batch_size']},log(mcap)=yes,lr={model_hparams['learning_rate']:.1e}", # key!
    'log_every_n_steps': 10,
    'save_top_k': 1,
    'val_check_interval': 1.0,

    # data size
    'precision': 32, # key!
    'overfit_batches': 0.0,
    'min_epochs': 10, # default: 10
    'max_epochs': 20, 
    'max_steps': None,
    'accumulate_grad_batches': 1,

    # Caution:
    # The check of patience depends on **how often you compute your val_loss** (`val_check_interval`). 
    # Say you check val every N baches, then `early_stop_callback` will compare to your latest N **baches**.
    # If you compute val_loss every N **epoches**, then `early_stop_callback` will compare to the latest N **epochs**.
    'early_stop_patience': 3,

    # Caution:
    # In pervious versions, if you check validatoin multiple times within a epoch,
    # you have to set `check_point_period=0`. However, starting from 1.0.7, even if 
    # you check validation multiples times within an epoch, you still need to set
    # `checkpoint_period=1`.
    'checkpoint_period': 1}

# delete all existing .ckpt files
refresh_ckpt()

# load split_df
split_df = load_split_df(data_hparams['window_size'])
    
# loop over windows
np.random.seed(trainer_hparams['seed'])
torch.manual_seed(trainer_hparams['seed'])

for yqtr in split_df.yqtr:
    
    # update current period
    data_hparams.update({'yqtr': yqtr})
    
    # train on select periods
    # if data_hparams['yqtr']=='2014-q1':
    train_one(Model, yqtr, data_hparams, model_hparams, trainer_hparams)
'''

# # RNN

'''
# ## Model

# CCGRU
class CCGRU(CC):
    def __init__(self, hparams):
        super().__init__(hparams)
        
        self.hparams = hparams
        
        # set model types
        self.task_type = 'single'
        self.feature_type = 'univariate'
        self.model_type = 'gru'
        self.attn_type = 'dotprod'
        self.text_in_dataset = True if self.feature_type!='fin-ratio' else False 
        
        # layers
        self.gru_expert = nn.GRU(hparams.d_model, hparams.rnn_hidden_size, num_layers=4, batch_first=True,
                                 dropout=0.1, bidirectional=True)
        self.dropout_expert = nn.Dropout(hparams.dropout)
        self.linear_car = nn.Linear(hparams.rnn_hidden_size*2, 1)

    # forward
    def forward(self, inp, valid_seq_len):
        # Note: inp is [N, S, E] and **already** been packed
        self.gru_expert.flatten_parameters()
        
        # if S is longer than `max_seq_len`, cut
        inp = inp[:,:self.hparams.max_seq_len,] # (N, S, E)
        valid_seq_len[valid_seq_len>self.hparams.max_seq_len] = self.hparams.max_seq_len # (N,)
        
        # RNN layers
        inp = pack_padded_sequence(inp, valid_seq_len, batch_first=True, enforce_sorted=False)
        x_expert = pad_packed_sequence(self.gru_expert(inp)[0], batch_first=True)[0][:,-1,:] # (N, E)
        
        # final FC layers
        y_car = self.linear_car(x_expert) # (N, E)
        
        return y_car
    
    # train step
    def training_step(self, batch, idx):
        
        car, transcriptid, embeddings, mask, alpha, car_m1_m1, car_m2_m2, car_m30_m3,\
        sest, sue, numest, sstdest, smedest, \
        mcap, roa, bm, debt_asset, volatility = batch
        
        # get valid seq_len
        valid_seq_len = torch.sum(~mask, -1)
        
        # forward
        y_car = self.forward(embeddings, valid_seq_len) # (N, 1)

        # compute loss
        loss_car = self.mse_loss(y_car, car.unsqueeze(-1)) # ()
        
        # logging
        return {'loss': loss_car, 'log': {'trainer_loss': loss_car}}
            
    # validation step
    def validation_step(self, batch, idx):
        
        car, transcriptid, embeddings, mask, alpha, car_m1_m1, car_m2_m2, car_m30_m3,\
        sest, sue, numest, sstdest, smedest, \
        mcap, roa, bm, debt_asset, volatility = batch
        
        # get valid seq_len
        valid_seq_len = torch.sum(~mask, -1)
        
        # forward
        y_car = self.forward(embeddings, valid_seq_len) # (N, 1)

        # compute loss
        loss_car = self.mse_loss(y_car, car.unsqueeze(-1)) # ()
        
        # logging
        return {'val_loss': loss_car}        
    
    # test step
    def test_step(self, batch, idx):
        
        car, transcriptid, embeddings, mask, alpha, car_m1_m1, car_m2_m2, car_m30_m3,\
        sest, sue, numest, sstdest, smedest, \
        mcap, roa, bm, debt_asset, volatility = batch
        
        # get valid seq_len
        valid_seq_len = torch.sum(~mask, -1)
        
        # forward
        y_car = self.forward(embeddings, valid_seq_len) # (N, 1)

        # compute loss
        loss_car = self.mse_loss(y_car, car.unsqueeze(-1)) # ()
        
        # logging
        return {'test_loss': loss_car}  


# + [markdown] toc-hr-collapsed=true toc-nb-collapsed=true
# ## run

# +
# loop over 24 windows
load_split_df()
load_targets()

# model hparams
model_hparams = {
    'preembedding_name': 'all_sbert_roberta_nlistsb_encoded', # key
    'batch_size': 8, # key
    'val_batch_size': 1, # key
    
    'max_seq_len': 1024, 
    'learning_rate': 3e-4,
    'task_weight': 1,

    'n_layers_encoder': 6,
    'n_head_encoder': 8, # optional
    'd_model': 1024,
    'rnn_hidden_size': 64,
    'final_tdim': 1024, # optional
    'dff': 2048,
    'attn_dropout': 0.1,
    'dropout': 0.5,
    'n_head_decoder': 8} # optional

# train hparams
trainer_hparams = {
    # checkpoint & log
    'note': 'temp',
    'checkpoint_path': 'D:\Checkpoints\earnings-call',
    'row_log_interval': 1,
    'save_top_k': 1,
    'val_check_interval': 0.25,

    # data size
    'overfit_pct': 1,
    'min_epochs': 0,
    'max_epochs': 1,
    'max_steps': None,
    'accumulate_grad_batches': 1,

    # Caution:
    # The check of patience depends on **how often you compute your val_loss** (`val_check_interval`). 
    # Say you check val every N baches, then `early_stop_callback` will compare to your latest N **baches**.
    # If you compute val_loss every N **epoches**, then `early_stop_callback` will compare to the latest N **epochs**.
    'early_stop_patience': 5,

    # Caution:
    # If set to 1, then save ckpt every 1 epoch
    # If set to 0, then save ckpt on every val!!! (if val improves)
    'checkpoint_period': 0}

# delete all existing .ckpt files
refresh_ckpt(trainer_hparams['checkpoint_path'])
    
# loop over 24!
for window_i in range(len(split_df)):
    # load preembeddings
    load_preembeddings(model_hparams['preembedding_name'])

    # train one window
    trainer_one(CCGRU, window_i, model_hparams, trainer_hparams)
'''

# # STL

# ## CCTransformerSTLTxt

# car ~ txt
class CCTransformerSTLTxt(CC):
    def __init__(self, d_model, learning_rate, attn_dropout, n_head_encoder, n_layers_encoder, dff, max_seq_len, model_type='STL', dropout=0.5):
        '''
        d_model: dimension of embedding. (default=1024)
        dff: fully-connected layer inside the transformer block. (default=2048)
        '''
        # `self.hparams` will be created by super().__init__
        super().__init__(learning_rate)
        
        self.save_hyperparameters()
        
        # positional encoding
        self.encoder_pos = PositionalEncoding(self.hparams.d_model, self.hparams.attn_dropout)
        
        # encoder layers for input, expert, nonexpert
        encoder_layers_expert = nn.TransformerEncoderLayer(self.hparams.d_model, self.hparams.n_head_encoder, self.hparams.dff, self.hparams.attn_dropout)
        
        # atten layers for CAR
        # self.attn_layers_car = nn.Linear(self.hparams.d_model, 1)
        # self.attn_dropout_1 = nn.Dropout(self.hparams.attn_dropout)
        
        # Build Encoder and Decoder
        self.encoder_expert = nn.TransformerEncoder(encoder_layers_expert, self.hparams.n_layers_encoder)
        
        # linear layer to produce final result
        self.fc_1 = nn.Linear(self.hparams.d_model, 1)
        # self.fc_2 = nn.Linear(32, 1)
        # self.dropout_1 = nn.Dropout(self.hparams.dropout)
        
    def forward(self):
        pass
    
    # forward
    def shared_step(self, batch):
        car, car_norm, inflow, inflow_norm, revision, revision_norm, \
        transcriptid, embeddings, src_key_padding_mask, \
        fin_ratios = batch
        
        # if S is longer than max_seq_len, cut
        embeddings = embeddings[:,:self.hparams.max_seq_len,] # (N, S, E)
        src_key_padding_mask = src_key_padding_mask[:,:self.hparams.max_seq_len] # (N, S)
        
        embeddings = embeddings.transpose(0, 1) # (S, N, E)
        
        # positional encoding
        x = self.encoder_pos(embeddings) # (S, N, E)
        
        # encode
        x_expert = self.encoder_expert(x, src_key_padding_mask=src_key_padding_mask).transpose(0,1) # (N, S, E)
        
        # decode with attn
        # x_attn = self.attn_dropout_1(F.softmax(self.attn_layers_car(x_expert), dim=1)) # (N, S, 1)
        # x_expert = torch.bmm(x_expert.transpose(-1,-2), x_attn).squeeze(-1) # (N, E)
        
        # decode with avgpool
        x_expert = x_expert.mean(1) # (N, E)
        
        # decode with maxpool
        # x_expert_maxpool = x_expert.max(1)[0] # (N, E)
        
        # concat
        # x_expert = torch.cat([x_expert_avgpool, x_expert_maxpool], dim=-1) # (N, 2E)

        # final FC
        y_car = self.fc_1(x_expert) # (N, 1)
        # y_car = self.fc_2(y_car)
        
        t_car = car_norm # (N,)
        
        # final output
        return transcriptid, y_car.squeeze(), t_car 
    
    # traning step
    def training_step(self, batch, idx):
        transcriptid, y_car, t_car = self.shared_step(batch)
        return {'y_car': y_car, 't_car': t_car}
        
    # validation step
    def validation_step(self, batch, idx):
        transcriptid, y_car, t_car = self.shared_step(batch)
        return {'y_car': y_car, 't_car': t_car}

    # test step
    def test_step(self, batch, idx):
        transcriptid, y_car, t_car = self.shared_step(batch)
        return {'transcriptid':transcriptid, 'y_car':y_car, 't_car': t_car}  


# ## CCTransformerSTLTxtFr

# car ~ txt + fr
class CCTransformerSTLTxtFr(CC):
    def __init__(self, d_model, learning_rate, attn_dropout, n_head_encoder, 
                 n_layers_encoder, dff, max_seq_len, model_type='STL', n_finratios=15, dropout=0.5):
        '''
        d_model: dimension of embedding. (default=1024)
        dff: fully-connected layer inside the transformer block. (default=2048)
        '''
        # `self.hparams` will be created by super().__init__
        super().__init__(learning_rate)
        
        self.save_hyperparameters()
        
        # positional encoding
        self.encoder_pos = PositionalEncoding(self.hparams.d_model, self.hparams.attn_dropout)
        
        # encoder layers for input, expert, nonexpert
        encoder_layers_expert = nn.TransformerEncoderLayer(self.hparams.d_model, self.hparams.n_head_encoder, self.hparams.dff, self.hparams.attn_dropout)
        
        # atten layers 
        # self.attn_layers_car = nn.Linear(self.hparams.d_model, 1)
        # self.attn_dropout_1 = nn.Dropout(self.hparams.attn_dropout)
        
        # Build Encoder and Decoder
        self.encoder_expert = nn.TransformerEncoder(encoder_layers_expert, self.hparams.n_layers_encoder)
        
        # linear layer to produce final result
        # txt_mixer_layer = TxtMixerLayer(self.hparams.d_model)
        # self.txt_mixer = FeatureMixer(txt_mixer_layer, self.hparams.n_layers_txtmixer)
        
        # fr_mixer_layers = FrMixerLayer(self.n_covariate)
        # self.fr_mixer = FeatureMixer(fr_mixer_layers, self.hparams.n_layers_frmixer)
        
        # final prediction layer
        # final_fc_mixer_layer = FeatureMixerLayer(self.hparams.d_model+self.n_covariate)
        # self.final_fc_mixer_layer = FeatureMixer(final_fc_mixer_layer, self.hparams.n_layers_finalfc)
        # self.fc_batchnorm = nn.BatchNorm1d(self.hparams.d_model+self.n_covariate)
        self.final_fc = nn.Linear(self.hparams.d_model+self.hparams.n_finratios, 1)
        
        # self.txt_fc_1 = nn.Linear(self.hparams.d_model, self.hparams.final_tdim)
        # self.txt_fc_2 = nn.Linear(self.hparams.d_model, self.hparams.final_tdim)
        # self.fc_1 = nn.Linear(self.hparams.final_tdim+self.n_covariate, self.hparams.final_tdim+self.n_covariate)
        # self.fc_2 = nn.Linear(self.hparams.final_tdim+self.n_covariate, self.hparams.final_tdim+self.n_covariate)
        # self.fc_3 = nn.Linear(self.hparams.final_tdim+self.n_covariate, 1)
        
        # dropout for final fc layers
        # self.txt_dropout_1 = nn.Dropout(self.hparams.dropout)
        # self.fc_dropout_1 = nn.Dropout(self.hparams.dropout)
        # self.fc_dropout_2 = nn.Dropout(self.hparams.dropout)
        # self.fc_dropout_3 = nn.Dropout(self.hparams.dropout) 
        
    def forward(self):
        pass
    
    def shared_step(self, batch):
        car, car_norm, inflow, inflow_norm, revision, revision_norm, \
        transcriptid, embeddings, src_key_padding_mask, \
        fin_ratios = batch
        
        # if S is longer than max_seq_len, cut
        embeddings = embeddings[:,:self.hparams.max_seq_len,] # (N, S, E)
        src_key_padding_mask = src_key_padding_mask[:,:self.hparams.max_seq_len] # (N, S)
        
        embeddings = embeddings.transpose(0, 1) # (S, N, E)
        
        # positional encoding
        x = self.encoder_pos(embeddings) # (S, N, E)
        
        # encode
        x_expert = self.encoder_expert(x, src_key_padding_mask=src_key_padding_mask).transpose(0,1) # (N, S, E)
        
        # decode with attn
        # x_attn = self.attn_dropout_1(F.softmax(self.attn_layers_car(x_expert), dim=1)) # (N, S, 1)
        # x_expert = torch.bmm(x_expert.transpose(-1,-2), x_attn).squeeze(-1) # (N, E)
        
        x_expert = x_expert.max(1)[0] # (N, E)
        
        
        # project text embedding to a lower dimension
        # x_expert = self.txt_dropout_1(F.relu(self.txt_fc_1(x_expert)))
        # x_expert = F.relu(self.txt_fc_2(x_expert))
        
        # x_expert = self.txt_mixer(x_expert)
        
        # Mix fin_ratios
        # fin_ratios = self.batch_norm(fin_ratios)
        # x_fr = self.fr_mixer(fin_ratios)
        
        # concate `x_final` with `fin_ratios`
        x_final = torch.cat([x_expert, fin_ratios], dim=-1) # (N, E+X) where X is the number of covariate (n_finratios)
        
        # final FC
        # x_final = self.fc_dropout_1(F.relu(self.fc_1(x_expert))) # (N, E+X)
        # x_car = self.final_fc_mixer_layer(x_final) # (N, E+X)
        y_car = self.final_fc(x_final)
        
        t_car = car_norm
        
        # final output
        return transcriptid.squeeze(), y_car.squeeze(), t_car.squeeze() 
    
    # traning step
    def training_step(self, batch, idx):
        transcriptid, y_car, t_car = self.shared_step(batch)
        return {'y_car': y_car, 't_car': t_car}
        
    # validation step
    def validation_step(self, batch, idx):
        transcriptid, y_car, t_car = self.shared_step(batch)
        return {'y_car': y_car, 't_car': t_car}

    # test step
    def test_step(self, batch, idx):
        transcriptid, y_car, t_car = self.shared_step(batch)
        return {'transcriptid':transcriptid, 'y_car':y_car, 't_car': t_car}  

# ## run

# +
# choose Model
Model = CCTransformerSTLTxt

# data hparams
data_hparams = {
    # inputs
    'preembedding_name': 'longformer', 
    'targets_name': 'f_sue_keydevid_car_finratio_vol_transcriptid_sim_inflow_revision_retail_sentiment_norm_outlier', 
    'tid_cid_pair_name': 'qa', 
    'tid_from_to_pair_name': '7qtr',
    
    # batch size
    'batch_size': 12,
    'val_batch_size':8,
    'test_batch_size':8,
    
    # window_size
    'text_in_dataset': True,
    'window_size': '2008-2017', # key!
}

# hparams
model_hparams = {
    'max_seq_len': 768, 
    'learning_rate':3e-4, # key!
    'n_layers_encoder': 4,
    'n_head_encoder': 8, 
    'd_model': 768,
    'dff': 2048, # default: 2048
    'attn_dropout': 0.1,
    # 'dropout': 0.5
} 

# train hparams
trainer_hparams = {
    # random seed
    'seed': 42,    # key
    
    # gpus
    'gpus': [0,1], # key

    # last: STL-57
    'machine': 'yu-workstation', # key!
    'note': f"STL-57", # key!
    'log_every_n_steps': 10,
    'save_top_k': 1,
    'val_check_interval': 0.2, # key! (Eg: 0.25 - check 4 times in a epoch)

    # epochs
    'precision': 32, # key!
    'overfit_batches': 0.0, # default 0.0. decimal or int
    'min_epochs': 3, # default: 3
    'max_epochs': 20, # default: 20
    'max_steps': 200, # default: None
    'accumulate_grad_batches': 1,

    # Caution:
    # The check of patience depends on **how often you compute your val_loss** (`val_check_interval`). 
    # Say you check val every N baches, then `early_stop_callback` will compare to your latest N **baches**.
    # If you compute val_loss every N **epoches**, then `early_stop_callback` will compare to the latest N **epochs**.
    'early_stop_patience': 8,

    # Caution:
    # In pervious versions, if you check validatoin multiple times within a epoch,
    # you have to set `check_point_period=0`. However, starting from 1.0.7, even if 
    # you check validation multiples times within an epoch, you still need to set
    # `checkpoint_period=1`.
    'checkpoint_period': 1}

# delete all existing .ckpt files
refresh_ckpt()

# load split_df
split_df = load_split_df(data_hparams['window_size'])

# load tid_cid_pair
# loop over windows!
for yqtr in split_df.yqtr:
    np.random.seed(trainer_hparams['seed'])
    torch.manual_seed(trainer_hparams['seed'])
    
    # Enforce yqtr>='2012-q4' (the earliest yqtr in window_size=='3y')
    # if yqtr == 'non-roll-01':

    # update current period
    data_hparams.update({'yqtr': yqtr})

    # train on select periods
    train_one(Model, yqtr, data_hparams, model_hparams, trainer_hparams)

# -

# # MTL

#
# ## model
#

# (MTL, hardshare) car + inf/rev ~ txt + fr
class CCTransformerMTLHard(CC):
    def __init__(self, hparams):
        # `self.hparams` will be created by super().__init__
        super().__init__(hparams)
        
        # check task weights sum to one
        assert self.hparams.car_weight+self.hparams.inflow_weight+self.hparams.revision_weight==1, 'car_weight + inflow_weight + revision_weight != 1'
        
        # specify model type
        self.model_type = 'TSFM'
        self.target_type = f'{self.hparams.car_weight:.2f}*car+{self.hparams.inflow_weight:.2f}*inf+{self.hparams.revision_weight:.2f}*rev' # key!
        self.feature_type = 'txt'    # key!
        self.emb_share = 'hard'      # key!
        self.normalize_target = True
        
        self.attn_type = 'dotprod'
        self.text_in_dataset = True if self.feature_type!='fr' else False 
        self.n_covariate = 15
        
        # positional encoding
        self.encoder_pos = PositionalEncoding(self.hparams.d_model, self.hparams.attn_dropout)
        
        # encoder layers for input, expert, nonexpert
        encoder_layers_expert = nn.TransformerEncoderLayer(self.hparams.d_model, self.hparams.n_head_encoder, self.hparams.dff, self.hparams.attn_dropout)
        
        # atten layers
        self.attn_layers_car = nn.Linear(self.hparams.d_model, 1)
        self.attn_dropout_1 = nn.Dropout(self.hparams.attn_dropout)
        
        # Build Encoder
        self.encoder_expert = nn.TransformerEncoder(encoder_layers_expert, self.hparams.n_layers_encoder)
        
        # linear layer to produce final result
        self.fc_car_1 = nn.Linear(self.hparams.d_model, 1)
        # self.fc_inflow_1 = nn.Linear(self.hparams.d_model, 1)
        self.fc_revision_1 = nn.Linear(self.hparams.d_model, 1)
        
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
        
        # aggregate with attn
        x_attn = self.attn_dropout_1(F.softmax(self.attn_layers_car(x_expert), dim=1)) # (N, S, 1)
        x_expert = torch.bmm(x_expert.transpose(-1,-2), x_attn).squeeze(-1) # (N, E)
        
        # final FC
        y_car = self.fc_car_1(x_expert) # (N, 1)
        # y_inflow = self.fc_inflow_1(x_expert) # (N,1)
        y_revision = self.fc_revision_1(x_expert)
        
        # log hist: x_expert
        if self.global_step%50==True:
            self.logger.experiment.log_histogram_3d(x_expert.detach().cpu().numpy(), name='feature:x_expert', step=self.global_step)
            # self.logger.experiment.log_histogram_3d(x_car.detach().cpu().numpy(), name='feature:x_car', step=self.global_step)
            # self.logger.experiment.log_histogram_3d(x_inflow.detach().cpu().numpy(), name='feature:x_inflow', step=self.global_step)
        
        # final output
        return y_car, y_revision
    
    # traning step
    def training_step(self, batch, idx):
        car, car_norm, inflow, inflow_norm, revision, revision_norm, \
        transcriptid, embeddings, mask, \
        fin_ratios = batch
        
        # decide if log activation histogram
        # log_histogram = True if idx%50==0 else False
        
        # forward
        y_car, y_revision = self.forward(embeddings, mask, fin_ratios) # (N, 1)
        
        # compute loss
        loss_car = self.mse_loss(y_car, car_norm.unsqueeze(-1)).unsqueeze(-1) # (1,)
        # loss_inflow = self.mse_loss(y_inflow, inflow_norm.unsqueeze(-1)).unsqueeze(-1) # (1,)
        loss_revision = self.mse_loss(y_revision, revision_norm.unsqueeze(-1)).unsqueeze(-1) # (1,)
        
        loss = self.hparams.car_weight*loss_car + self.hparams.revision_weight*loss_revision
        
        # logging
        return {'loss': loss, 'log': {'trainer_loss': loss}}
        
    # validation step
    def validation_step(self, batch, idx):

        car, car_norm, inflow, inflow_norm, revision, revision_norm, \
        transcriptid, embeddings, mask, \
        fin_ratios = batch
        
        # forward
        y_car, y_revision = self.forward(embeddings, mask, fin_ratios) # (N, 1)
        
        # compute loss
        loss_car = self.mse_loss(y_car, car_norm.unsqueeze(-1)).unsqueeze(-1) # (1,)
        # loss_inflow = self.mse_loss(y_inflow, inflow_norm.unsqueeze(-1)).unsqueeze(-1) # (1,)
        loss_revision = self.mse_loss(y_revision, revision_norm.unsqueeze(-1)).unsqueeze(-1) # (1,)
        
        loss = self.hparams.car_weight*loss_car + self.hparams.revision_weight*loss_revision
        
        # logging
        # return {'val_loss': loss, 'val_loss_car': loss_car, 'val_loss_inflow': loss_inflow}
        return {'val_loss': loss, 'val_loss_car': loss_car, 'val_loss_revision': loss_revision}

    # test step
    def test_step(self, batch, idx):
        car, car_norm, inflow, inflow_norm, revision, revision_norm, \
        transcriptid, embeddings, mask, \
        fin_ratios = batch
        
        # forward
        y_car, y_revision = self.forward(embeddings, mask, fin_ratios) # (N, 1)
        
        # compute loss
        loss_car = self.mse_loss(y_car, car_norm.unsqueeze(-1)).unsqueeze(-1) # (1,)
        # loss_inflow = self.mse_loss(y_inflow, inflow_norm.unsqueeze(-1)).unsqueeze(-1) # (1,)
        loss_revision = self.mse_loss(y_revision, revision_norm.unsqueeze(-1)).unsqueeze(-1) # (1,)
        
        loss = self.hparams.car_weight*loss_car + self.hparams.revision_weight*loss_revision
        
        # logging
        # return {'test_loss': loss, 'test_loss_car': loss_car, 'test_loss_inflow': loss_inflow}  
        return {'test_loss': loss, 'test_loss_car': loss_car, 'test_loss_revision': loss_revision}  


# (MTL, softshare) x*car + (1-x)*inf ~ txt + fr
class CCTransformerMTLSoft(CC):
    def __init__(self, hparams):
        # `self.hparams` will be created by super().__init__
        super().__init__(hparams)
        
        # specify model type
        self.model_type = 'TSFM'
        self.target_type = 'car+inf'
        self.feature_type = 'txt+fr'
        self.emb_share = 'hard'
        self.normalize_target = True
        
        self.attn_type = 'dotprod'
        self.text_in_dataset = True if self.feature_type!='fr' else False 
        self.n_covariate = 15
        
        # positional encoding
        self.encoder_pos = PositionalEncoding(self.hparams.d_model, self.hparams.attn_dropout)
        
        # encoder layers for input, expert, nonexpert
        encoder_layers_expert = nn.TransformerEncoderLayer(self.hparams.d_model, self.hparams.n_head_encoder, self.hparams.dff, self.hparams.attn_dropout)
        
        # atten layers
        self.attn_layers_car = nn.Linear(self.hparams.d_model, 1)
        self.attn_dropout_1 = nn.Dropout(self.hparams.attn_dropout)
        
        # Build Encoder
        self.encoder_expert = nn.TransformerEncoder(encoder_layers_expert, self.hparams.n_layers_encoder)
        
        # linear layer to produce final result
        self.linear_car_1 = nn.Linear(self.hparams.d_model, self.hparams.d_model)
        self.linear_car_2 = nn.Linear(self.hparams.d_model, self.hparams.final_tdim)
        self.linear_car_3 = nn.Linear(self.hparams.final_tdim+self.n_covariate, self.hparams.final_tdim+self.n_covariate)
        self.linear_car_4 = nn.Linear(self.hparams.final_tdim+self.n_covariate, self.hparams.final_tdim+self.n_covariate)
        self.linear_car_5 = nn.Linear(self.hparams.final_tdim+self.n_covariate, 1)
        
        self.linear_inflow = nn.Linear(self.hparams.final_tdim, 1)
        # self.linear_revision = nn.Linear(hparam.final_tdim, 1)
        
        # dropout for final fc layers
        self.final_dropout_1 = nn.Dropout(self.hparams.dropout)
        self.final_dropout_2 = nn.Dropout(self.hparams.dropout)
        self.final_dropout_3 = nn.Dropout(self.hparams.dropout)
        
        # layer normalization
        if self.hparams.normalize_layer:
            self.layer_norm = nn.LayerNorm(self.hparams.final_tdim+self.n_covariate)
            
        # batch normalization
        if self.hparams.normalize_batch:
            self.batch_norm = nn.BatchNorm1d(self.n_covariate)

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
        
        # multiply with attn
        x_attn = self.attn_dropout_1(F.softmax(self.attn_layers_car(x_expert), dim=1)) # (N, S, 1)
        x_expert = torch.bmm(x_expert.transpose(-1,-2), x_attn).squeeze(-1) # (N, E)
        
        # mix with covariate
        x_expert = self.final_dropout_1(F.relu(self.linear_car_1(x_expert))) # (N, E)
        x_expert = F.relu(self.linear_car_2(x_expert)) # (N, final_tdim)
        
        # batch normalization
        if self.hparams.normalize_batch:
            fin_ratio = self.batch_norm(fin_ratios)
        
        x_car = torch.cat([x_expert, fin_ratios], dim=-1) # (N, X + final_tdim) where X is the number of covariate (n_covariate)

            
        # final FC
        y_inflow = self.linear_inflow(x_expert)
        # y_revision = self.linear_revision(x_expert)
        
        x_car = self.final_dropout_2(F.relu(self.linear_car_3(x_car))) # (N, X + final_tdim)
        y_car = self.linear_car_5(x_car) # (N,1)
        
        # final output
        return y_car, y_inflow
    
    # traning step
    def training_step(self, batch, idx):
        car, car_norm, inflow, inflow_norm, revision, revision_norm, \
        transcriptid, embeddings, mask, \
        fin_ratios = batch
        
        # forward
        y_car, y_inflow = self.forward(embeddings, mask, fin_ratios) # (N, 1)
        
        # compute loss
        loss_car = self.mse_loss(y_car, car_norm.unsqueeze(-1)).unsqueeze(-1) # (1,)
        loss_inflow = self.mse_loss(y_inflow, inflow_norm.unsqueeze(-1)).unsqueeze(-1) # (1,)
        
        
        assert self.hparams.car_weight+self.hparams.inflow_weight==1, 'car_weight + inflow_weight != 1'
        
        loss = self.hparams.car_weight*loss_car + self.hparams.inflow_weight*loss_inflow
        
        # logging
        return {'loss': loss, 'log': {'trainer_loss': loss}}
        
    # validation step
    def validation_step(self, batch, idx):
        car, car_norm, inflow, inflow_norm, revision, revision_norm, \
        transcriptid, embeddings, mask, \
        fin_ratios = batch
        
        # forward
        y_car, y_inflow = self.forward(embeddings, mask, fin_ratios) # (N, 1)
        
        # compute loss
        loss_car = self.mse_loss(y_car, car_norm.unsqueeze(-1)).unsqueeze(-1) # (1,)
        loss_inflow = self.mse_loss(y_inflow, inflow_norm.unsqueeze(-1)).unsqueeze(-1) # (1,)
        
        loss = loss_car + loss_inflow
        
        # logging
        return {'val_loss': loss, 'val_loss_car': loss_car, 'val_loss_inflow': loss_inflow}

    # test step
    def test_step(self, batch, idx):
        car, car_norm, inflow, inflow_norm, revision, revision_norm, \
        transcriptid, embeddings, mask, \
        fin_ratios = batch
        
        # forward
        y_car, y_inflow = self.forward(embeddings, mask, fin_ratios) # (N, 1)
        
        # compute loss
        loss_car = self.mse_loss(y_car, car_norm.unsqueeze(-1)).unsqueeze(-1) # (1,)
        loss_inflow = self.mse_loss(y_inflow, inflow_norm.unsqueeze(-1)).unsqueeze(-1) # (1,)
        
        loss = loss_car + loss_inflow

        # logging
        return {'test_loss': loss, 'test_loss_car': loss_car, 'test_loss_inflow': loss_inflow} 


# ## run

# + endofcell="--"
'''
# choose Model
Model = CCTransformerSTLTxtFr

# hparams
model_hparams = {
    'seed': 42, # key!
    'preembedding_name': 'all_sbert_roberta_nlistsb_encoded', # key!
    'targets_name': 'f_sue_keydevid_car_finratio_vol_transcriptid_sim_inflow_revision_sentiment_text_norm_wsrz', # key!
    'window_size': '3y',  # key!
    'gpus': [0,1],
    
    # task weight
    'car_weight': 1,      # Key!
    'inflow_weight': 0,   # key!
    'revision_weight': 0, # key!
    
    'batch_size': 28,     # key!
    'val_batch_size': 28,
    'max_seq_len': 768, 
    'learning_rate':1e-4, # key!
    'task_weight': 1,
    'n_layers_encoder': 4,
    'n_layers_txtmixer': 2, # key!
    'n_layers_frmixer': 2,  # key!
    'n_layers_finalfc': 2,  # key!
    'n_head_encoder': 8, 
    'd_model': 1024,
    'final_tdim': 1024, # key!
    'dff': 2048,
    'attn_dropout': 0.1,
    'dropout': 0.5,
    'n_head_decoder': 8} 

trainer_hparams = {
    # log
    'machine': 'yu-workstation',  # key!
    'note': f"STL-37,(car~txt+fr 3y BatchNormFr BatchNormTxt),txtMixer=2({model_hparams['final_tdim']}),fcMixer=2,standCAR=yes,standFr=no,bsz={model_hparams['batch_size']},seed={model_hparams['seed']},log(mcap)=yes,lr={model_hparams['learning_rate']:.2g}", # key!
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
load_split_df(model_hparams['window_size'])
    
# load targets_df
load_targets(model_hparams['targets_name'])

# load preembeddings
load_preembeddings(model_hparams['preembedding_name'])
    
# loop over 24!
np.random.seed(model_hparams['seed'])
torch.manual_seed(model_hparams['seed'])

for window_i in range(len(split_df)):

    # train one window
    trainer_one(Model, window_i, model_hparams, trainer_hparams)
'''
# -
# --
