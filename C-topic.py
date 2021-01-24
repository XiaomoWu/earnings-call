# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Init

# %%
import datatable as dt
import multiprocessing as mp
import numpy as np
import os
import spacy
import time

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datatable import f
from functools import partial
from spacy.tokens import Doc, DocBin, Span
from tqdm.auto import tqdm

dt.init_styles()

WORK_DIR = '/home/yu/OneDrive/CC'
DATA_DIR = f'{WORK_DIR}/data'
os.chdir(WORK_DIR)

exec(open('/home/yu/OneDrive/App/Settings/jupyter + R + Python/python_startup.py').read())

# %% [markdown]
# # Convert Doc to "text tokens"

# %%
# Load DocBin from disk
nlp = spacy.load('en_core_web_lg')
docs = list(DocBin(store_user_data=True).from_disk('data/doc_sp500_test.spacy').get_docs(nlp.vocab))

# register extension for Doc
Doc.set_extension('transcriptid', default=None, force=True)

# Register extension for Span
Span.set_extension('transcriptid', default=None, force=True)
Span.set_extension('componentid', default=None, force=True)
Span.set_extension('componentorder', default=None, force=True)
Span.set_extension('componenttypeid', default=None, force=True)
Span.set_extension('speakerid', default=None, force=True)
Span.set_extension('speakertypeid', default=None, force=True)
Span.set_extension('is_component', default=False, force=True)


# %%
# Select componentid that belongs to MD and QA
ld('text_component_sp500', ldname='text_component')
text_component = dt.Frame(text_component)

# componentid: Management Discussion
componentids_md = text_component[(f.transcriptcomponenttypeid==2) & (f.speakertypeid==2), f.transcriptcomponentid].to_list()[0]

# componentid: Q & A
componentids_qa = text_component[((f.transcriptcomponenttypeid==3) | (f.transcriptcomponenttypeid==4)) & ((f.speakertypeid==2)|(f.speakertypeid==3)), f.transcriptcomponentid].to_list()[0]


# %%
# Convert Doc to "text tokens"

# Filtering Rule:
# - only keep lemma
# - no stop words (stop words is informative while comparing)
# - no punctuation
# - no "like number"

def make_text_tokens(docs, componentids):
    texttoken = {}
    
    # For every doc, join the required spans into a list of str
    for doc in tqdm(docs):
        txttok = []
        for span in doc.spans['components']:
            if span._.componentid in componentids:
                txttok.extend([t.lower_ for t in span 
                if ((not t.is_punct) & (not t.like_num) & (not t.is_stop))])

        # If no text found, add an empty str
        if len(txttok)==0:
            txttok = ['']

        # return
        texttoken[doc._.transcriptid] = txttok
    
    return texttoken

# start = time.perf_counter()
# texttoken_md = make_text_tokens(docs[:1000], componentids_md)
# gap = time.perf_counter() - start
# print(f'{gap} secs')

# texttoken_qa = make_text_tokens(docs, componentids_qa)


# %%
# print(texttoken_md[108])


# %%
# Convert Doc to "text tokens"

# Filtering Rule:
# - only keep lemma
# - no stop words (stop words is informative while comparing)
# - no punctuation
# - no "like number"

def make_text_tokens(doc):
    
    # For every doc, join the required spans into a list of str
    txttok = []
    for span in doc.spans['components']:
        if span._.componentid in componentids_md:
            txttok.extend([t.lower_ for t in span 
            if ((not t.is_punct) & (not t.like_num) & (not t.is_stop))])

    # If no text found, add an empty str
    if len(txttok)==0:
        txttok = ['']

    # return 
    return {doc._.transcriptid: txttok}

start = time.perf_counter()

with mp.Pool(5) as pool:
    results = pool.imap_unordered(make_text_tokens, docs[:100], chunksize=10)

gap = time.perf_counter() - start
print(f'{gap} secs')

# with ProcessPoolExecutor(max_workers=8) as executor:
#     results = executor.map(make_text_tokens, docs[:5], chunksize=1)

# texttoken_md = make_text_tokens(docs, componentids_md)
# texttoken_qa = make_text_tokens(docs, componentids_qa)

# %% [markdown]
# # Convert to DTM

# %%
'''
from sklearn.feature_extraction.text import CountVectorizer

# Convert to DTM

# Setting:
# - keep ALL tokens
vectorizer = CountVectorizer(preprocessor=lambda x: x,
                             tokenizer=lambda x: x,
                             lowercase=False,
                             ngram_range=(1,2))

# Learn vocabulary 
vectorizer.fit(texttoken_md.values())
vectorizer.fit(texttoken_qa.values())

# Make DTM
dtm_md = vectorizer.transform(texttoken_md.values())
dtm_qa = vectorizer.transform(texttoken_qa.values())

# Compute similarity
from sklearn.metrics.pairwise import cosine_similarity
similarity = np.diag(cosine_similarity(dtm_md, dtm_qa))
similarity

'''


