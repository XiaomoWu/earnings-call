#!/usr/bin/env python
import pandas as pd
import subprocess

# Project directory
DATA_DIR = '/home/yu/OneDrive/CC/data'

# Model config
window_size = '6y'

# Run model
split_df = pd.read_csv(f'{DATA_DIR}/split_dates.csv')
split_df = split_df.loc[split_df.window_size==window_size]

# for yqtr in split_df.yqtr:
for yqtr in ['2018-q2']:
    subprocess.run(['python', 'C-model.py', '--yqtr', yqtr, '--window_size', window_size])