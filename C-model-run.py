#!/usr/bin/env python
import pandas as pd
import os
import subprocess
import shutil


# run startup file
with open("/home/yu/OneDrive/App/Settings/jupyter + R + Python/python_startup.py", 'r') as _:
    exec(_.read())

# Project directory
DATA_DIR = '/home/yu/OneDrive/CC/data'

# Model config
window_size = '7y'
note = 'MTL-31'

# Run model
split_df = pd.read_csv(f'{DATA_DIR}/split_dates.csv')
split_df = split_df.loc[split_df.window_size==window_size]

# copy script
run_script = f'C-model-{note}.py.tmp'
shutil.copyfile('C-model.py', run_script)

# train
for yqtr in split_df.yqtr:
# for yqtr in ['2015-q2']:
    # if yqtr<'2015-q2':
        # continue

    subprocess.run(['python', run_script, '--yqtr', yqtr, '--window_size', window_size, '--note', note])

# delete run_script finally
os.unlink(run_script)