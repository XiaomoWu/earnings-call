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
#     display_name: Python-3.8
#     language: python
#     name: python3
# ---

# # Init

# +
# import tensorflow as tf
import argparse
import comet_ml
import datatable as dt
import gc
import glob
import numpy as np
import torch
import os
import pytorch_lightning as pl
import torch.nn.functional as F
import torch.optim as optim
import shutil
import pandas as pd
import pyarrow.feather as feather
import warnings

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
CHECKPOINT_ARCHIVE_DIR = f'{CHECKPOINT_DIR}/archive'

# COMET API KEY
COMET_API_KEY = 'tOoHzzV1S039683RxEr2Hl9PX'

# set random seed
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

# +
names = os.listdir('/home/yu/OneDrive/CC/data/Embeddings/longformer')

len(names)
len(set(names))
# -

preemb_dir = '/home/yu/OneDrive/CC/data/Embeddings/longformer'
for name in names:
    if not re.search('\d+\.pt', name):
        os.unlink(f'{preemb_dir}/{name}')


