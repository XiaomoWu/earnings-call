# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-toolsai.jupyter added
import os
try:
	os.chdir(os.path.join(os.getcwd(), '../../../../tmp/f694e69e-4651-416f-b4b7-f98b6a7a34a5'))
	print(os.getcwd())
except:
	pass
# %% [markdown]
# # Init

# %%
import datatable as dt
import spacy

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datatable import f
from spacy.attrs import ORTH
from spacy.tokens import Doc, DocBin, Span
from tqdm.auto import tqdm

# set working directory
WORK_DIR = '/home/yu/OneDrive/CC'
DATA_DIR = '/home/yu/OneDrive/CC/data'

os.chdir(WORK_DIR)

# initialize data.table
dt.init_styles()


# %%
# Helpers
import logging
import pandas as pd
import pickle
import pyarrow.feather as feather
import re
import time

# creat a logger
logger = logging.getLogger()

def ld(filename:str, ldname=None, ldtype=None, path='./data', force=False):
    starttime = time.perf_counter()

    # check if ldtype is valid
    # possible value: NULL, "rds", "feather"
    if ldtype!=None and ldtype not in ['pkl', 'feather']:
        raise Exception('`ldtype` must be one of "pkl", "feather" or "None"')

    # Verify file type: rds or feather
    # if file doesn't exist, stop;
    # if both exists, stop and ask for clarification;
    # if only one exists, assign it to `lddir`
    hit = [p for p in os.listdir(path) if re.search(f'{filename}\.(rds|feather)', p)]
    hit_extensions = unique([h.split('.')[-1] for h in hit])

    if len(hit)==0:
        raise Exception(f'Cannot find {filename} with extension ({hit_extensions})!')
    elif len(hit)==1:
        lddir = f'{path}/{hit[0]}'
        ldtype = hit[0].split('.')[-1]
        filename_ext = hit[0]
    elif len(hit)==2 and ldtype!=None:
        lddir = f'{path}/{filename}/{ldtype}'
        filename_ext = f'{filename}/{ldtype}'
    else:
        raise Exception(f'Multiple extensions of "{filename}" found ({str(hit_extensions)[1:-1]}), please clarify!')

    # get file size before loading
    file_size = neat_file_size(lddir)

    # If force is False and filename/ldname already exists in globals(), SKIP
    if force==False and (filename in globals() or ldname in globals()):
        file_in_env = filename if ldname==None else f'"{filename}" or "{ldname}"'

        print(f'{file_in_env} ({file_size}) already loaded, will NOT load again!')
    
    # else, laod the file
    else:
        # first, load as val
        if ldtype=='feather':
            val = feather.read_feather(lddir)
        elif ldtype=='pkl':
            with open(lddir, 'wb') as f:
                pickle.load(f)

        # get time elapsed
        elapsed_time = time.perf_counter() - starttime

        # then assign `val` a name
        # if ldname is None, use filename as ldname
        if ldname==None:
            globals()[filename] = val
            print(f'"{filename_ext}" ({file_size}) loaded ({pretty_time_delta(elapsed_time)})')
        else:
            globals()[ldname] = val
            print(f'"{filename_ext}" ({file_size}) loaded as "{ldname}" ({pretty_time_delta(elapsed_time)})')

def unique(seq: 'tuple or list') -> list:
    '''remove duplicate while keeping original order
    '''
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

def pretty_time_delta(seconds):
    seconds = int(seconds)
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if days > 0:
        return f'{days}days {hours}h {minutes}m {seconds}s'
    elif hours > 0:
        return f'{hours}h {minutes}m {seconds}s'
    elif minutes > 0:
        return f'{minutes}m {seconds}s'
    elif seconds > 0:
        return f'{seconds}s'
    else:
        return f'<1s'

def neat_file_size(file_path):
    '''Return file size (as string) in human-readable format
    '''
    bytes = os.path.getsize(file_path)
    if bytes >= 1024**3:
        gbs = bytes/1024 ^ 3
        return f'{gbs:.1f} GB'
    elif 1024**2 < bytes & bytes <= 1024**3:
        mbs = bytes/1024**2
        return f'{mbs:.1f} MB'
    elif 1024 < bytes & bytes <= 1024**2:
        kbs = bytes/1024
        return f'{kbs:.1f} KB'
    else:
        bs = bytes
        return f'{bs:.1f} B'

# %% [markdown]
# # Build Doc in spaCy
# %% [markdown]
# ## Register Extension

# %%
# Use the simpliest pipeline
# 'tok2vec', 'parser', 'lemmatizer', 'tagger', 'attribute_ruler'
nlp = spacy.load("en_core_web_lg", disable=['ner'])

'''
# Add a simple sentencizer
nlp.add_pipe('sentencizer')

# add [EOC] as special case in tokenization
special_case = [{ORTH: "[EOC]"}]
nlp.tokenizer.add_special_case("[EOC]", special_case)
'''

# register extension for Span
Span.set_extension('transcriptid', default=None, force=True)
Span.set_extension('componentid', default=None, force=True)
Span.set_extension('componentorder', default=None, force=True)
Span.set_extension('componenttypeid', default=None, force=True)
Span.set_extension('speakerid', default=None, force=True)
Span.set_extension('speakertypeid', default=None, force=True)
Span.set_extension('is_component', default=False, force=True)

# %% [markdown]
# ## Load data

# %%
# Load components as a 2D table
ld('text_component_sp500', ldname='text_component', force=True)
text_component = dt.Frame(text_component)

# conver 2D table to tuples
text_component = text_component.to_tuples()
text_component = [(line[6], 
                   {'transcriptid': line[2],
                    'componentid': line[0],
                    'componenttypeid': line[4],
                    'componentorder': line[3],
                    'speakerid': int(line[5]) if line[5]!=None else None,
                    'speakertypeid': int(line[1]) if line[1]!=None else None
                   }) for line in text_component]

text_component_grouped = {}
for text, context in text_component:
    tid = context['transcriptid']
    if tid in text_component_grouped:
        text_component_grouped[tid].append((text, context))
    else:
        text_component_grouped[tid] = [(text, context)]

# %% [markdown]
# ## Build Doc
# %% [markdown]
# > Only need to run this sectoin **ONCE**. It will hold all the ground truth and will never be altered.

# %%
# Final output holder
docs = []

# Iterate through every transcriptid
for line in tqdm(text_component_grouped.values(), total=len(text_component_grouped)):

    # Output holder
    components = []

    # Within every transcriptid, iterature through every component
    for component, context in nlp.pipe(line, as_tuples=True):
        
        # Assign component-level attributes
        component[:]._.is_component = True
        component[:]._.transcriptid = context['transcriptid']
        component[:]._.componentid = context['componentid']
        component[:]._.componenttypeid = context['componenttypeid']
        component[:]._.componentorder = context['componentorder']
        component[:]._.speakerid = context['speakerid']
        component[:]._.speakertypeid = context['speakertypeid']

        # Assign sentence-level attributes
        for sent in component.sents:
            sent._.componentid = context['componentid']

        # return
        components.append(component)

    # join components into one Doc
    doc = Doc.from_docs(components)

    # create SpanGroup "components" for doc
    spans_component = []
    for k, v in doc.user_data.items():
        if k[1]=='is_component':
            if v==True:
                spans_component.append(doc.char_span(k[2], k[3]))

    doc.spans['components'] = spans_component 

    # return     
    docs.append(doc)

DocBin(store_user_data=True, docs=docs).to_disk('data/doc_sp500.spacy')

# %% [markdown]
# ## Save DocBin
# %% [markdown]
# > **Warnings**
# 
# > When saving DocBin, you must also save the nlp object for recovery.

# %%
DocBin(store_user_data=True, docs=docs).to_disk('data/doc_sp500.spacy')


