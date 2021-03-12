#!/usr/bin/env python
import argparse
import pandas as pd
import os
import subprocess
import shutil

from argparse import Namespace

# run startup file
with open("/home/yu/OneDrive/App/Settings/jupyter + R + Python/python_startup.py", 'r') as _:
    exec(_.read())

# Project directory
DATA_DIR = '/home/yu/OneDrive/CC/data'

# ---------------------------------------
# Parse args 
# ---------------------------------------

# ----- comment below when debugging -----------------
parser = argparse.ArgumentParser(description='Earnings Call')
parser.add_argument('-n', '--n_workers', default=1, type=int, required=False)
parser.add_argument('-i', '--worker_id', default=1, type=int, required=True)
args = parser.parse_args()
# ----- comment above when debugging -----------------

# uncomment the following line when debugging
# args = Namespace(**{'n_workers': 1, 'worker_id': 1})

# ---------------------------------------
# Model config 
# ---------------------------------------
window_size = '7y'
note = 'MTLTxt-17'
filter_yqtrs = []


# ---------------------------------------
# Run 
# ---------------------------------------
split_df = pd.read_csv(f'{DATA_DIR}/split_dates.csv')
split_df = split_df.loc[split_df.window_size==window_size]

# split all the windows into n_parts parts
n_workers = args.n_workers
worker_id = args.worker_id # [1, n_worker]

yqtrs = ntile(split_df.yqtr.tolist(), ntiles=n_workers)[worker_id-1]
if len(filter_yqtrs)>=1:
    yqtrs = list(set(yqtrs) & set(filter_yqtrs))
print(f'Created {len(yqtrs)} yqtrs for worker {worker_id}/{n_workers}: {yqtrs}')

# copy script
run_script = f'C-model-{note}-worker{worker_id}.py'
shutil.copyfile('C-model.py', run_script)
# run_script = 'C-model.py'

# train
for yqtr in yqtrs:
    subprocess.run(['python', run_script, '--yqtr', yqtr, '--window_size', window_size, '--note', note])

# delete run_script finally
os.unlink(run_script)
