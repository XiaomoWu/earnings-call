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
CHECKPOINT_TEMP_DIR = f'{CHECKPOINT_DIR}/archive'

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
    move all `.ckpt` files to `/archive`
    '''
    # create ckpt_dir if not exists
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
    
    # create ckpt_temp_dir if not exists
    if not os.path.exists(CHECKPOINT_TEMP_DIR):
        os.makedirs(CHECKPOINT_TEMP_DIR)
    
    for name in os.listdir(CHECKPOINT_DIR):
        if name.endswith('.ckpt'):
            shutil.move(f'{CHECKPOINT_DIR}/{name}', f'{CHECKPOINT_TEMP_DIR}/{name}')

# helpers: load targets
def load_targets(targets_name, force=False):
    targets_df = feather.read_feather(f'{DATA_DIR}/{targets_name}.feather')
    targets_df = targets_df[targets_df.outlier_flag1==False]
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
    Always compare to the same model: 'ols: car_stand ~ fr'
    then log to Comet
    '''
    performance = dt.Frame(pd.read_feather('data/performance_by_yqtr.feather'))


    ols_rmse = performance[(f.model_name=='ols: car_stand ~ fr') & (f.window_size==window_size) & (f.yqtr==yqtr), f.rmse][0,0]
    logger.experiment.log_parameter('ols_rmse', ols_rmse)
    
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
    
    def __init__(self, yqtr, split_type, text_in_dataset, window_size, targets_df, split_df, tid_cid_pair=None, tid_from_to_pair=None, preembeddings=None):
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
            # targets_df = targets_df.iloc[:int(len(targets_df)*0.9)]
            
        elif split_type=='val':
            targets_df = targets_df[targets_df.ciq_call_date.between(train_start, train_end)].sample(frac=1, random_state=42)
            targets_df = targets_df.iloc[int(len(targets_df)*0.9):]

        elif split_type=='test':
            targets_df = targets_df[targets_df.ciq_call_date.between(test_start, test_end)]

        
        if text_in_dataset:
            # make sure targets_df only contains transcriptid that're also 
            # in preembeddings
            targets_df = targets_df.loc[targets_df.transcriptid.isin(set(preembeddings.keys()))]
            
            self.tid_cid_pair = tid_cid_pair
            self.tid_from_to_pair = tid_from_to_pair
            self.preembeddings = preembeddings
        
        # Assign states
        self.text_in_dataset = text_in_dataset

        self.targets_df = targets_df
        self.train_start = train_start
        self.train_end = train_end
        self.test_start = test_start
        self.test_end = test_end
        self.split_type = split_type

        
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
        car_0_30_stand = targets.car_0_30_stand
        revision = targets.revision
        revision_stand = targets.revision_stand
        inflow = targets.inflow
        inflow_stand = targets.inflow_stand
        
        # using the normalized features
        similarity = targets.similarity_bigram_stand
        sentiment = targets.qa_positive_sent_stand
        sue = targets.sue_stand
        sest = targets.sest_stand        
        alpha = targets.alpha_stand
        volatility = targets.volatility_stand
        mcap = targets.mcap_stand
        bm = targets.bm_stand
        roa = targets.roa_stand
        debt_asset = targets.debt_asset_stand
        numest = targets.numest_stand
        smedest = targets.smedest_stand
        sstdest = targets.sstdest_stand
        car_m1_m1 = targets.car_m1_m1_stand
        car_m2_m2 = targets.car_m2_m2_stand
        car_m30_m3 = targets.car_m30_m3_stand
        volume = targets.volume_stand

        if self.text_in_dataset:
            # inputs: preembeddings
            embeddings = assemble_embedding(transcriptid, 
                                            self.preembeddings,
                                            self.tid_cid_pair,
                                            self.tid_from_to_pair)

            return car_0_30, car_0_30_stand, inflow, inflow_stand, revision,\
                   revision_stand,  transcriptid, embeddings, \
                   [alpha, car_m1_m1, car_m2_m2, car_m30_m3, sest, sue, numest, sstdest, smedest, mcap, roa, bm, debt_asset, volatility, volume]
        else:
            return torch.tensor(transcriptid,dtype=torch.int64), \
                   torch.tensor(car_0_30,dtype=torch.float32), \
                   torch.tensor(car_0_30_stand,dtype=torch.float32), \
                   torch.tensor(inflow,dtype=torch.float32), \
                   torch.tensor(inflow_stand,dtype=torch.float32), \
                   torch.tensor(revision,dtype=torch.float32), \
                   torch.tensor(revision_stand,dtype=torch.float32), \
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
    def __init__(self, yqtr, targets_name, batch_size, val_batch_size,
                 test_batch_size, text_in_dataset, window_size, 
                 preembedding_name=None, tid_cid_pair_name=None,
                 tid_from_to_pair_name=None):
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
        if self.text_in_dataset:
            global preembeddings
            load_preembeddings(self.preembedding_name)
            tid_cid_pair = load_tid_cid_pair(self.tid_cid_pair_name)
            tid_from_to_pair = load_tid_from_to_pair(self.tid_from_to_pair_name)
        else:
            preembeddings = None
            tid_cid_pair = None
            tid_from_to_pair = None
            
        targets_df = load_targets(self.targets_name)
        split_df = load_split_df(self.window_size)

        
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
        car_0_30, car_0_30_stand, inflow, inflow_stand, revision, revision_stand, \
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
               torch.tensor(car_0_30_stand, dtype=torch.float32), \
               torch.tensor(inflow, dtype=torch.float32), \
               torch.tensor(inflow_stand, dtype=torch.float32), \
               torch.tensor(revision, dtype=torch.float32), \
               torch.tensor(revision_stand, dtype=torch.float32), \
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
        
        # save & log `y_car`
        y_car_filename = f'{DATA_DIR}/y_car.feather'
        
        df = pd.DataFrame({'transcriptid':transcriptid.to('cpu'),
                           'y_car':y_car.to('cpu'),
                           't_car':t_car.to('cpu')})
        feather.write_feather(df, y_car_filename)
            
        self.logger.experiment.log_asset(y_car_filename)
            
    # optimizer
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer   
    
# Model: MTL
class CCMTL(CC):
    '''Mainly define the `*_step_end` methods
    '''
    # loss
    def binary_cross_entropy_loss(self, y, t):
        return F.binary_cross_entropy(y, t)
        
    # def training_step_end
    def training_step_end(self, outputs):
        y_car = outputs['y_car']
        y_rev = outputs['y_rev']
        y_inf = outputs['y_inf']
        
        t_car = outputs['t_car']
        t_rev = outputs['t_rev']
        t_inf = outputs['t_inf']
        
        loss_car = self.mse_loss(y_car, t_car)
        loss_rev = self.mse_loss(y_rev, t_rev)
        loss_inf = self.mse_loss(y_inf, t_inf)
        
        loss = loss_car + self.hparams.alpha*(loss_rev + loss_inf)/2
        
        return {'loss':loss}
    
    # def validation_step_end
    def validation_step_end(self, outputs):
        y_car = outputs['y_car']
        y_rev = outputs['y_rev']
        y_inf = outputs['y_inf']
        
        t_car = outputs['t_car']
        t_rev = outputs['t_rev']
        t_inf = outputs['t_inf']
        
        return {'y_car': y_car, 'y_rev': y_rev, 'y_inf': y_inf,
                't_car': t_car, 't_rev': t_rev, 't_inf': t_inf}
        
    # validation step
    def validation_epoch_end(self, outputs):
        '''
        outputs: a list. len(outputs) == num_steps.
            e.g., outputs = [{'val_loss': 3}, {'val_loss': 4}]
        '''
        y_car = torch.cat([x['y_car'] for x in outputs])
        y_rev = torch.cat([x['y_rev'] for x in outputs])
        y_inf = torch.cat([x['y_inf'] for x in outputs])
        
        t_car = torch.cat([x['t_car'] for x in outputs])
        t_rev = torch.cat([x['t_rev'] for x in outputs])
        t_inf = torch.cat([x['t_inf'] for x in outputs])
        
        loss_car = self.mse_loss(y_car, t_car)
        loss_rev = self.mse_loss(y_rev, t_rev)
        loss_inf = self.mse_loss(y_inf, t_inf)
        
        loss = loss_car + self.hparams.alpha*(loss_rev + loss_inf)/2
        
        rmse = torch.sqrt(loss)
        self.log('val_rmse', rmse, on_step=False)
        
    # test step
    def test_step_end(self, outputs):
        transcriptid = outputs['transcriptid']
        
        y_car = outputs['y_car']
        y_rev = outputs['y_rev']
        y_inf = outputs['y_inf']
        
        t_car = outputs['t_car']
        t_rev = outputs['t_rev']
        t_inf = outputs['t_inf']
        
        return {'transcriptid': transcriptid,
                'y_car': y_car, 'y_rev': y_rev, 'y_inf': y_inf,
                't_car': t_car, 't_rev': t_rev, 't_inf': t_inf}
    
    def test_epoch_end(self, outputs):
        
        transcriptid = torch.cat([x['transcriptid'] for x in outputs])
        y_car = torch.cat([x['y_car'] for x in outputs])
        y_rev = torch.cat([x['y_rev'] for x in outputs])
        y_inf = torch.cat([x['y_inf'] for x in outputs])
        
        t_car = torch.cat([x['t_car'] for x in outputs])
        t_rev = torch.cat([x['t_rev'] for x in outputs])
        t_inf = torch.cat([x['t_inf'] for x in outputs])
        
        loss_car = self.mse_loss(y_car, t_car)
        loss_rev = self.mse_loss(y_rev, t_rev)
        loss_inf = self.mse_loss(y_inf, t_inf)
        
        loss = loss_car
        
        rmse = torch.sqrt(loss_car)
        self.log('test_rmse', rmse, on_step=False)
        
        # save & log `y_car`
        y_car_filename = f'{DATA_DIR}/y_car.feather'
        
        df = pd.DataFrame({'transcriptid':transcriptid.to('cpu'),
                           'y_car':y_car.to('cpu'),
                           'y_rev':y_rev.to('cpu'),
                           'y_inf':y_inf.to('cpu'),
                           't_car':t_car.to('cpu'),
                           't_rev':t_rev.to('cpu'),
                           't_inf':t_inf.to('cpu')})
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
    ckpt_prefix = f"{trainer_hparams['note']}_{data_hparams['yqtr']}".replace('*', '')
    ckpt_prefix = ckpt_prefix + '_{epoch}'
    
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        verbose=True,
        mode='min',
        monitor='val_rmse',
        dirpath=CHECKPOINT_DIR,
        filename=ckpt_prefix,
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
                         accelerator='dp',
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
    # log_ols_rmse(logger, data_hparams['yqtr'], data_hparams['window_size'])
    
    # upload test_start
    log_test_start(logger, data_hparams['window_size'], data_hparams['yqtr'])
    
    # If run on ASU, upload code explicitly
    if trainer_hparams['machine'] == 'ASU':
        codefile = [name for name in os.listdir('.') if name.endswith('.py')]
        assert len(codefile)==1, f'There must be only one `.py` file in the current directory! {len(codefile)} files detected: {codefile}'
        logger.experiment.log_asset(codefile[0])
    
    
    # refresh GPU memory
    # refresh_cuda_memory()

    
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
        # refresh_cuda_memory()
        logger.finalize('finished')


# + [markdown] toc-hr-collapsed=true toc-hr-collapsed=true toc-hr-collapsed=true toc-hr-collapsed=true toc-hr-collapsed=true toc-hr-collapsed=true
# # MLP
# -

# ## model

# MLP
class CCMLP(CC):
    def __init__(self, learning_rate, dropout, model_type='MLP'):
        super().__init__(learning_rate)
        
        self.save_hyperparameters()
        
        # dropout layers
        self.dropout_1 = nn.Dropout(self.hparams.dropout)
        # self.dropout_2 = nn.Dropout(self.hparams.dropout)
        
        # fc layers
        self.fc_1 = nn.Linear(15, 16)
        self.fc_2 = nn.Linear(16, 1)
        #self.fc_3 = nn.Linear(32, 1)
        
    def forward(self):
        pass
    
    def shared_step(self, batch):
        transcriptid, car, car_stand, inflow, inflow_stand, \
        revision, revision_stand, manual_txt, fin_ratios = batch
        # x = torch.cat([fin_ratios, manual_txt], dim=-1) # (N, 2+15)
    
        x = fin_ratios
        
        x_car = self.dropout_1(F.relu(self.fc_1(x)))
        y_car = self.fc_2(x_car) # (N, 1)    
        
        t_car = car_stand
        
        return transcriptid, y_car.squeeze(), t_car 
        
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
    'targets_name': 'targets_final', # key!

    'batch_size': 64,
    'val_batch_size':64,
    'test_batch_size':64,
    
    'text_in_dataset': False,
    'window_size': '6y', # key!
}

# model hparams
model_hparams = {
    'learning_rate': 1e-4,
    'dropout': 0.1,
}

# train hparams
trainer_hparams = {
    # random seed
    'seed': 42,    # key
    
    # gpus
    'gpus': [0,1], # key

    # checkpoint & log
    
    # last: 
    'machine': 'yu-workstation', # key!
    'note': f"MLP-02", # key!
    'log_every_n_steps': 10,
    'save_top_k': 1,
    'val_check_interval': 1.0,

    # data size
    'precision': 32, # key!
    'overfit_batches': 0.0,
    'min_epochs': 10, # default: 10
    'max_epochs': 500, # default: 20. Must be larger enough to have at least one "val_rmse is not in the top 1"
    'max_steps': None, # default None
    'accumulate_grad_batches': 1,

    # Caution:
    # The check of patience depends on **how often you compute your val_loss** (`val_check_interval`). 
    # Say you check val every N baches, then `early_stop_callback` will compare to your latest N **baches**.
    # If you compute val_loss every N **epoches**, then `early_stop_callback` will compare to the latest N **epochs**.
    'early_stop_patience': 10, # default: 3

    # Caution:
    # In pervious versions, if you check validatoin multiple times within a epoch,
    # you have to set `check_point_period=0`. However, starting from 1.0.7, even if 
    # you check validation multiples times within an epoch, you still need to set
    # `checkpoint_period=1`.
    'checkpoint_period': 1} # default 1

# delete all existing .ckpt files
refresh_ckpt()

# load split_df
split_df = load_split_df(data_hparams['window_size'])
    
# loop over windows
np.random.seed(trainer_hparams['seed'])
torch.manual_seed(trainer_hparams['seed'])

for yqtr in split_df.yqtr:

    # Only test after 2012-q4
    if yqtr>='2012-q4':

        # update current period
        data_hparams.update({'yqtr': yqtr})

        # train on select periods
        train_one(Model, yqtr, data_hparams, model_hparams, trainer_hparams)
        
'''

# + [markdown] toc-hr-collapsed=true toc-hr-collapsed=true toc-hr-collapsed=true toc-hr-collapsed=true toc-hr-collapsed=true toc-hr-collapsed=true toc-hr-collapsed=true
# # RNN
# -

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

# + [markdown] toc-hr-collapsed=true toc-hr-collapsed=true toc-hr-collapsed=true toc-hr-collapsed=true toc-hr-collapsed=true toc-hr-collapsed=true toc-hr-collapsed=true
# # STL
# -

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
        car, car_stand, inflow, inflow_stand, revision, revision_stand, \
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
        
        t_car = car_stand # (N,)
        
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
        car, car_stand, inflow, inflow_stand, revision, revision_stand, \
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
        # fin_ratios = self.batch_stand(fin_ratios)
        # x_fr = self.fr_mixer(fin_ratios)
        
        # concate `x_final` with `fin_ratios`
        x_final = torch.cat([x_expert, fin_ratios], dim=-1) # (N, E+X) where X is the number of covariate (n_finratios)
        
        # final FC
        # x_final = self.fc_dropout_1(F.relu(self.fc_1(x_expert))) # (N, E+X)
        # x_car = self.final_fc_mixer_layer(x_final) # (N, E+X)
        y_car = self.final_fc(x_final)
        
        t_car = car_stand
        
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

'''
# choose Model
Model = CCTransformerSTLTxt

# data hparams
data_hparams = {
    # inputs
    'preembedding_name': 'longformer', 
    'targets_name': 'f_sue_keydevid_car_finratio_vol_transcriptid_sim_inflow_revision_retail_sentiment_stand_outlier', 
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
    'max_steps': None, # default: None
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
'''


# # MTL

# ## CCMTLFr

# MLP
class CCMTLFr(CCMTL):
    def __init__(self, learning_rate, dropout, alpha, model_type='MTL'):
        super().__init__(learning_rate)
        
        self.save_hyperparameters()
        
        # dropout layers
        self.dropout_1 = nn.Dropout(self.hparams.dropout)
        # self.dropout_2 = nn.Dropout(self.hparams.dropout)
        
        # fc layers
        self.fc_1 = nn.Linear(15, 16)
        # self.fc_2 = nn.Linear(16, 16)
        self.fc_car = nn.Linear(16, 1)
        self.fc_rev = nn.Linear(16, 1)
        self.fc_inf = nn.Linear(16, 1)
        
    def forward(self, batch):
        transcriptid, car, car_stand, inflow, inflow_stand, \
        revision, revision_stand, manual_txt, fin_ratios = batch
        
        t_car = car_stand
        t_rev = revision_stand
        t_inf = inflow_stand

        x = fin_ratios
        x = F.relu(self.fc_1(x))
        # x = F.relu(self.fc_2(x))
        
        return transcriptid, t_car, x
    
    
    def shared_step(self, batch):
        transcriptid, car, car_stand, inflow, inflow_stand, \
        revision, revision_stand, manual_txt, fin_ratios = batch
        
        t_car = car_stand
        t_rev = revision_stand
        t_inf = inflow_stand

        # x = torch.cat([fin_ratios, manual_txt], dim=-1) # (N, 2+15)
        x = fin_ratios
        
        x = self.dropout_1(F.relu(self.fc_1(x)))
        # x = self.dropout_2(F.relu(self.fc_2(x)))
        y_car = self.fc_car(x) # (N, 1)    
        y_rev = self.fc_rev(x) # (N, 1)
        y_inf = self.fc_inf(x) # (N, 1)
        
        return transcriptid, y_car.squeeze(), y_rev.squeeze(), \
               y_inf.squeeze(), t_car, t_rev, t_inf 
        
    # train step
    def training_step(self, batch, idx):
        transcriptid, y_car, y_rev, y_inf, \
        t_car, t_rev, t_inf = self.shared_step(batch)
        return {'y_car': y_car, 'y_rev': y_rev, 'y_inf': y_inf,
                't_car': t_car, 't_rev': t_rev, 't_inf': t_inf}
        
    # validation step
    def validation_step(self, batch, idx):
        transcriptid, y_car, y_rev, y_inf, \
        t_car, t_rev, t_inf = self.shared_step(batch)
        return {'y_car': y_car, 'y_rev': y_rev, 'y_inf': y_inf,
                't_car': t_car, 't_rev': t_rev, 't_inf': t_inf}
        
    # test step
    def test_step(self, batch, idx):
        transcriptid, y_car, y_rev, y_inf, \
        t_car, t_rev, t_inf = self.shared_step(batch)
        return {'transcriptid': transcriptid,
                'y_car': y_car, 'y_rev': y_rev, 'y_inf': y_inf,
                't_car': t_car, 't_rev': t_rev, 't_inf': t_inf}

# ## run

# +
# '''
# choose Model
Model = CCMTLFr

# data hparams
data_hparams = {
    'targets_name': 'targets_final', # key!

    'batch_size': 64,
    'val_batch_size':64,
    'test_batch_size':64,
    
    'text_in_dataset': False,
    'window_size': '5y', # key!
}

# model hparams
model_hparams = {
    'alpha': 0.1, # key!
    'learning_rate': 1e-4,
    'dropout': 0.1, # default: 0.5
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
    'note': f"MTL-16", # key!
    'log_every_n_steps': 10,
    'save_top_k': 1,
    'val_check_interval': 1.0,

    # data size
    'precision': 32, # key!
    'overfit_batches': 0.0,
    'min_epochs': 10, # default: 10
    'max_epochs': 500, # default: 20. Must be larger enough to have at least one "val_rmse is not in the top 1"
    'max_steps': None, # default None
    'accumulate_grad_batches': 1,

    # Caution:
    # The check of patience depends on **how often you compute your val_loss** (`val_check_interval`). 
    # Say you check val every N baches, then `early_stop_callback` will compare to your latest N **baches**.
    # If you compute val_loss every N **epoches**, then `early_stop_callback` will compare to the latest N **epochs**.
    'early_stop_patience': 10, # default: 3

    # Caution:
    # In pervious versions, if you check validatoin multiple times within a epoch,
    # you have to set `check_point_period=0`. However, starting from 1.0.7, even if 
    # you check validation multiples times within an epoch, you still need to set
    # `checkpoint_period=1`.
    'checkpoint_period': 1} # default 1

# delete all existing .ckpt files
refresh_ckpt()

# load split_df
split_df = load_split_df(data_hparams['window_size'])
    
# loop over windows
np.random.seed(trainer_hparams['seed'])
torch.manual_seed(trainer_hparams['seed'])

print(f'Start training...{trainer_hparams["note"]}')

for yqtr in split_df.yqtr:

    # Only test after 2012-q4
    if yqtr == '2013-q4':

        # update current period
        data_hparams.update({'yqtr': yqtr})

        # train on select periods
        train_one(Model, yqtr, data_hparams, model_hparams, trainer_hparams)
# '''
# -

# # Extractor

# ## cRT with OLS

'''
# ----------------------------
# Specify Model and data
# ----------------------------
Model = CCMTLFr

ckpt_name = 'MTL-14'
ckpt_paths = [path for path in os.listdir(f'{CHECKPOINT_TEMP_DIR}')
              if path.startswith(ckpt_name+'_')]
print(f'N checkpoint found: {len(ckpt_paths)}')

# load data
data_hparams = {
    'targets_name': 'targets_final', # key!

    'batch_size': 64,
    'val_batch_size':64,
    'test_batch_size':64,
    
    'text_in_dataset': False,
    'window_size': '6y', # key!
}

# ----------------------------
# Extract
# ----------------------------
def extract(model, dataloader):
    # Extract y, x using model
    
    with torch.no_grad():
        transcriptid = []
        features = []
        t_car = []

        for batch in dataloader:
            tid, car, f = model.forward(batch)
            transcriptid.append(tid)
            t_car.append(car)
            features.append(f)

        transcriptid = dt.Frame(transcriptid=torch.cat(transcriptid).numpy())
        t_car = dt.Frame(t_car=torch.cat(t_car).numpy())
        features = dt.Frame(torch.cat(features, dim=0).numpy())
        features.names = [n.replace('C', 'feature') for n in features.names]

        targets = dt.cbind([transcriptid, t_car, features])

        t, x = dmatrices(f't_car ~ {"+".join(features.names)}',
                         data=targets,
                         return_type='dataframe')
        
        return transcriptid['transcriptid'].to_list()[0], t, x

split_df = load_split_df(data_hparams['window_size'])

yt_extractor = []
for yqtr in tqdm(split_df.yqtr):
    
    if yqtr<'2012-q4':
        continue
        
    # load train/test data
    data_hparams.update({'yqtr': yqtr})

    datamodule = CCDataModule(**data_hparams)
    datamodule.setup()
    
    train_dataloader = datamodule.train_dataloader()
    test_dataloader = datamodule.test_dataloader()
    
    # load model
    ckpt_path = [path for path in ckpt_paths 
                 if path.startswith(f'{ckpt_name}_{yqtr}')]
    assert len(ckpt_path)==1, f'Multiple or no checkpoint found for "{ckpt_name}_{yqtr}"'
    ckpt_path = ckpt_path[0]
    
    model = Model.load_from_checkpoint(f'{CHECKPOINT_TEMP_DIR}/{ckpt_path}')
    model.eval()
    
    # extract
    import statsmodels.api as sm
    from patsy import dmatrices

    _, t_train, x_train = extract(model, train_dataloader)
    transcriptid, t_test, x_test = extract(model, test_dataloader)
        
    # Fit OLS on Train
    fitted = sm.OLS(t_train, x_train).fit()
    
    # Apply OLS on Test
    y_test = fitted.predict(x_test).to_list()
    t_test = t_test['t_car'].to_list()
    
    df = dt.Frame(transcriptid=transcriptid,
                  t_car=t_test,
                  y_car=y_test)
    df[:, update(model_name=ckpt_name+'_extractor',
                 window_size=data_hparams['window_size'],
                 yqtr=yqtr)]
    
    yt_extractor.append(df)
    
yt_extractor = dt.rbind(yt_extractor)

# Combine
ld('yt_extractor', 'old_yt_extractor', force=True)
all_yt_extractor = dt.rbind([yt_extractor, old_yt_extractor])

sv('all_yt_extractor', 'yt_extractor')

'''

# ## cRT with MLP

'''
# ----------------------------
# Specify Model and data
# ----------------------------
class cRT(CC):
    def __init__(self, learning_rate, extractor):
        super().__init__(learning_rate)
        
        self.extractor = extractor
        self.fc_1 = nn.Linear(16,1)
        self.dropout_1 = nn.Dropout(0.1)
        
    def shared_step(self, batch):
        transcriptid, t_car, x = self.extractor(batch)
        y = self.dropout_1(F.relu(self.fc_1(x)))
        return transcriptid, t_car, y
        
    def forward(self, batch):
        transcriptid, t_car, y = self.shared_step(batch) 
        return transcriptid, t_car, y
        
    def training_step(self, batch, idx):
        transcriptid, t_car, y = self.shared_step(batch)
        return {'y':y, 't':t_car, 'transcriptid':transcriptid}
    
    def training_step_end(self, outputs):
        transcriptid = outputs['transcriptid']
        y = outputs['y']
        t = outputs['t']
        
        loss = self.mse_loss(y,t)
        return {'loss':loss}
        

ckpt_name = 'MTL-11'
ckpt_paths = [path for path in os.listdir(f'{CHECKPOINT_TEMP_DIR}')
              if path.startswith(ckpt_name+'_')]
print(f'N checkpoint found: {len(ckpt_paths)}')

# load data
data_hparams = {
    'targets_name': 'targets_final', # key!

    'batch_size': 64,
    'val_batch_size':64,
    'test_batch_size':64,
    
    'text_in_dataset': False,
    'window_size': '6y', # key!
}

# ----------------------------
# Extract
# ----------------------------
def predict(model, dataloader):
    model.eval()

    with torch.no_grad():
        transcriptid = []
        y_car = []
        t_car = []

        for batch in dataloader:
            tid, t, y = model.forward(batch)
            transcriptid.append(tid)
            t_car.append(t)
            y_car.append(y)
            
        transcriptid = dt.Frame(transcriptid=torch.cat(transcriptid).numpy())
        t_car = dt.Frame(t_car=torch.cat(t_car).numpy())
        y_car = dt.Frame(torch.cat(y_car, dim=0).numpy())

        targets = dt.cbind([transcriptid, t_car, features])

        
        return targets

split_df = load_split_df(data_hparams['window_size'])

yt_extractor = []
for yqtr in tqdm(split_df.yqtr):
    
    if yqtr<'2012-q4':
        continue
        
    # load model
    ckpt_path = [path for path in ckpt_paths 
                 if path.startswith(f'{ckpt_name}_{yqtr}')]
    assert len(ckpt_path)==1, \
           f'Multiple or no checkpoint found for "{ckpt_name}_{yqtr}"'
    ckpt_path = ckpt_path[0]
    
    extractor = CCMTLFr.load_from_checkpoint(f'{CHECKPOINT_TEMP_DIR}/{ckpt_path}')
    extractor.eval()
        
    # load train/test data
    data_hparams.update({'yqtr': yqtr})

    datamodule = CCDataModule(**data_hparams)
    datamodule.setup()
    test_dataloader = datamodule.test_dataloader()
    
    # Retrain classifier
    classrt = cRT(learning_rate=1e-4,
                  extractor=extractor)
    logger = CometLogger(
        api_key=COMET_API_KEY,
        save_dir='/data/logs',
        project_name='earnings-call',
        experiment_name=data_hparams['yqtr'],
        workspace='amiao',
        display_summary_level=0)
    logger.experiment.log_parameters({'note':'cRT-test'})
    trainer = pl.Trainer(gpus=[1], accelerator='dp',
                         max_epochs=20)
    trainer.fit(classrt, datamodule)
    
    # predict
    df = predict(classrt, test_dataloader)
        
    df[:, update(model_name=ckpt_name+'_fcextractor',
                 window_size=data_hparams['window_size'],
                 yqtr=yqtr)]
    
    yt_extractor.append(df)
    
yt_extractor = dt.rbind(yt_extractor)

# Combine
# ld('yt_extractor', 'old_yt_extractor', force=True)
# all_yt_extractor = dt.rbind([yt_extractor, old_yt_extractor])

# sv('all_yt_extractor', 'yt_extractor')
'''
