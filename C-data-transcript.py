# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
import datatable as dt
import os
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

exec(open('/home/yu/OneDrive/App/Settings/jupyter + R + Python/python_startup.py').read())


# %% [markdown]
# # Build Doc in spaCy
# %% [markdown]
# ## Register Extension

# %%
# Use the simpliest pipeline
# 'tok2vec', 'parser', 'lemmatizer', 'tagger', 'attribute_ruler'
# nlp = spacy.load("en_core_web_sm", disable=['ner', 'parser', 'tok2vec', 'tagger', 'lemmatizer', 'attribute_ruler'])
nlp = spacy.load("en_core_web_lg")

# Add a simple sentencizer
# nlp.add_pipe('sentencizer')

# register extension for Doc
Doc.set_extension('transcriptid', default=None, force=True)

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
        
del text_component

# %% [markdown]
# ## Build Doc
# %% [markdown]
# > Only need to run this sectoin **ONCE**. It will hold all the ground truth and will never be altered.

# %%
# Final output holder

# Iterate through every transcriptid
def make_one_doc(line):
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

    # Add Doc-level attribute: "transcriptid"
    doc._.transcriptid = context['transcriptid']

    # create SpanGroup "components" for doc
    spans_component = []
    for k, v in doc.user_data.items():
        if k[1]=='is_component':
            if v==True:
                spans_component.append(doc.char_span(k[2], k[3]))

    doc.spans['components'] = spans_component 

    # return     
    # docs.append(doc)
    return doc

'''
# ------------- Without Chunks ----------------
with ProcessPoolExecutor(16) as executor:
    docs = list(tqdm(executor.map(make_one_doc, text_component_grouped.values(), chunksize=200), total=len(text_component_grouped)))

# save as DocBin
# Note: put these lines OUTSIDE of the ProcessPoolExecutor context
docbin = DocBin(store_user_data=True, docs=docs, attrs=['ORTH', 'POS', 'ENT_IOB', 'ENT_TYPE', 'LEMMA'])

del text_component_grouped, text_component_grouped

docbin.to_disk(f'data/doc_sp500_trf.spacy')
'''

# ------------- With Chunks -------------------
# Because of memory limitation, we split the data into chunks and process/store one by one.

n_chunks = 5
chunk_size = len(text_component_grouped)//n_chunks+1

text_component_grouped_chunked = list(chunks(list(text_component_grouped.values()), chunk_size))

del text_component_grouped

for i in range(n_chunks):

    print(f'Processing chunks: {i+1}/{n_chunks}')

    data = text_component_grouped_chunked[i]

    # process using pools
    with ProcessPoolExecutor(16) as executor:
        docs = list(tqdm(executor.map(make_one_doc, data, chunksize=200), total=len(data)))

    docbin = DocBin(store_user_data=True, docs=docs, attrs=['ORTH', 'LEMMA', 'MORPH', 'POS', 'TAG', 'HEAD', 'DEP', 'ENT_IOB', 'ENT_TYPE'])
    
    docbin.to_disk(f'data/doc_sp500_lg_{i}.spacy')

    del docs, docbin, data

# %% [markdown]
# ## Save DocBin
# %% [markdown]
# > **Warnings**
# 
# > When saving DocBin, you must also save the nlp object for recovery.

# %%
# DocBin(store_user_data=True, docs=docs).to_disk('data/doc_sp500_test.spacy')
# print('DocBin Saved!')

