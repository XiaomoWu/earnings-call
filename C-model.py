# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.4.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Init

# %%
# import tensorflow as tf
import comet_ml
import spacy
import sentence_transformers
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
import shutil
from pytorch_memlab import LineProfiler
from collections import OrderedDict, defaultdict
from spacy.lang.en import English
from argparse import Namespace
from scipy.sparse import coo_matrix

from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data.dataset import random_split
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from transformers import BertModel, BertTokenizer, GPT2Model, GPT2Tokenizer, RobertaTokenizer, RobertaModel, XLNetTokenizer, XLNetModel
from sentence_transformers import SentenceTransformer

# working directory
ROOT_DIR = 'C:/Users/rossz/OneDrive/CC'
DATA_DIR = f'{ROOT_DIR}/data'
print(f'ROOT_DIR: {ROOT_DIR}')
print(f'DATA_DIR: {DATA_DIR}')

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

# %% [markdown] toc-hr-collapsed=true toc-nb-collapsed=true toc-hr-collapsed=true toc-nb-collapsed=true toc-hr-collapsed=true toc-nb-collapsed=true toc-hr-collapsed=true toc-nb-collapsed=true toc-hr-collapsed=true toc-nb-collapsed=true toc-hr-collapsed=true toc-nb-collapsed=true toc-hr-collapsed=true toc-nb-collapsed=true toc-hr-collapsed=true toc-nb-collapsed=true toc-hr-collapsed=true toc-nb-collapsed=true toc-hr-collapsed=true toc-nb-collapsed=true toc-hr-collapsed=true toc-nb-collapsed=true toc-hr-collapsed=true toc-nb-collapsed=true toc-hr-collapsed=true toc-nb-collapsed=true toc-hr-collapsed=true toc-nb-collapsed=true toc-hr-collapsed=true toc-nb-collapsed=true
# # Pre-encode

# %%
model_path = "C:/Users/rossz/.cache/torch/sentence_transformers/public.ukp.informatik.tu-darmstadt.de_reimers_sentence-transformers_v0.2_roberta-large-nli-stsb-mean-tokens.zip"

with open(os.path.join(model_path, 'modules.json')) as fIn:
    contained_modules = json.load(fIn)
    
sbert_modules = OrderedDict()
for module_config in contained_modules:
    module_class = sentence_transformers.util.import_from_string(module_config['type'])
    module = module_class.load(os.path.join(model_path, module_config['path']))
    sbert_modules[module_config['name']] = module
    
# For Roberta, pad_token_id == 1
if 'roberta' in model_path:
    sbert_pad_token_id = 1
else:
    raise Exception("You're not using RoBERTa, double check your pad_token_id")
    
    
sbert_model = nn.Sequential(sbert_modules)
log_gpu_memory()

# %% [markdown] toc-hr-collapsed=true toc-nb-collapsed=true
# ## gpt2

# %% [markdown]
# ### sentencize

# %%
# load (X,Y)
df = pd.read_feather(f'{DATA_DIR}/f_sue_keydevid_car_transcriptid_textpresent.feather')
print(f'num of calls: {len(df)}')

# %%
# spacy model
nlp = English()  
sentencizer = nlp.create_pipe("sentencizer")
nlp.add_pipe(sentencizer)

def sentencize(df:str, sentencizer, text_field):
    transcriptids = df['transcriptid']
    texts = df[text_field]
    
    res = []
    for tid, text in tqdm(zip(transcriptids, texts), total=len(texts)):
        sents = [sent.text for sent in sentencizer(text).sents] 
        for sid, sent in enumerate(sents):
            res.append((tid, sid, sent))
    return res

text_present_sentencized = sentencize(df, nlp, 'text_present')

# %%
pd.DataFrame(text_present_sentencized, columns=['transcriptid', 'sentenceid', 'text']).to_feather(f'{DATA_DIR}/text_present_sentencized.feather')

# %% [markdown]
# ### memory limit test

# %% [markdown]
# > Get sentence length

# %%
seq_len_present = []
sents = pd.read_feather(f'{DATA_DIR}/text_present_sentencized_nochunk.feather')['text'].tolist()

for sent in tqdm(sents):
    seq_len_present.append(len(gpt_tokenizer.encode(sent, add_special_tokens=True)))
sv('seq_len_present')

# %% [markdown]
# > Find `seq_len` limit

# %%
batch_size = 96
seq_len = 256
pad_token_id = 50257

with torch.no_grad():
    for i in tqdm(range(30)):
        inputs = torch.tensor(list(torch.utils.data.RandomSampler(range(pad_token_id), replacement=True,num_samples=batch_size*seq_len))).reshape(batch_size, seq_len)
        y = gpt_model(inputs)[0].to(cpu)

# %% [markdown]
# ### load `gpt`

# %% [markdown]
# > Load Transformers
# - gpt-2 doesn't have <PAD> tokens, so we have to manually add it
# - `pad_token_id`==50257

# %%
# BERT model
# bert_tokenizer = BertTokenizer.from_pretrained('bert-large-cased-whole-word-masking')
# bert_model = BertModel.from_pretrained('bert-large-cased-whole-word-masking')

# GPT-2
gpt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
gpt_tokenizer.add_special_tokens({'pad_token': '<PAD>'})
pad_token_id = gpt_tokenizer.pad_token_id

gpt_model = GPT2Model.from_pretrained('gpt2-medium')
gpt_model.resize_token_embeddings(len(gpt_tokenizer))
gpt_model.eval() # enable eval model
gpt_model = nn.DataParallel(gpt_model)
gpt_model.to(cuda0); # load model to GPU

# cpu (live in CPU)
cpu_tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
cpu_tokenizer.add_special_tokens({'pad_token': '<PAD>'})
cpu_pad_token_id = cpu_tokenizer.pad_token_id

cpu_model = GPT2Model.from_pretrained('gpt2-medium')
cpu_model.resize_token_embeddings(len(cpu_tokenizer))
cpu_model.eval() # enable eval model

log_gpu_memory();


# %% [markdown]
# ### create Dataset

# %% [markdown]
# > Task: Create Dataset
# >
# > Warnings:
# - The output of Dataset transformer *CANNOT* be empty. For empty sentence (of length 0), we need pad with`pad_token_id` 

# %%
#-------------------- Create Dataset -------------------
class Tokenize():
    def __init__(self, tokenizer, pad_token_id, max_seq_len):
        '''
        max_seq_len: There're still ass-cover statement in the call, which are very long.
            I remove every sentence which are longer than `max_seq_len`
        pad_token_id: for empty sentences, set length to 1 and fill with `pad_token_id`
        '''
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        
    def __call__(self, sample):
        transcriptid, sentenceid, sent = sample
        sent = self.tokenizer.encode(sent, add_special_tokens=True)
        if len(sent) == 0 or len(sent) < self.max_seq_len:
            return transcriptid, sentenceid, torch.tensor([self.pad_token_id], dtype=torch.long)
        else:
            sent = torch.tensor(sent, dtype=torch.long)
            return transcriptid, sentenceid, sent


class CCDataset(Dataset):
    def __init__(self, df, transform=None):
        '''
        Args:
            df: DataFrame 
        '''
        self.transform = transform
        self.df = df
        self.length_sorted_idx = np.argsort([len(sent) for sent in df['text'].tolist()])

        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        # sample: (transcripid, sentenceid, text)
        sample = tuple(self.df.iloc[self.length_sorted_idx[idx]])
            
        if self.transform:
            sample = self.transform(sample)
            
        return sample


# %% [markdown]
# ### create DataLoader

# %%
# --------------------------- Create DataLoader--------------------------
def collate_fn(data: list):
    transcriptids, sentenceids, sents = list(zip(*data))
    
    # valid seq_len
    valid_seq_len = torch.tensor([sent.shape[0] for sent in sents])
    
    # pad
    sents_padded = pad_sequence(sents, batch_first=True, padding_value=pad_token_id)
    
    # create mask for padded sentences
    # (n_seq_per_batch, max_seq_len)
    mask = (sents_padded != pad_token_id).int().float()

    return transcriptids, sentenceids, sents_padded, mask, valid_seq_len


# %% [markdown]
# ### encode

# %%
text_present_sentencized = pd.read_feather(f'{DATA_DIR}/text_present_sentencized.feather')
text_present_sentencized.head()


# %%
# ------------------------- Excution -----------------------------
def pre_encode(dl, model, save_path, start):
    with torch.no_grad():
        res = []
        for i_batch, (transcriptid, sentenceid, text, mask, valid_seq_len) in enumerate(tqdm(dl)):
            
            assert len(transcriptid) == valid_seq_len.shape[0]
            
            output = model(text, attention_mask=mask)[0].to(cpu)
            batch_size = output.shape[0]
            d_model = output.shape[-1]

            # for every doc in a batch, do mask average pooling
            for i, end in enumerate(valid_seq_len):
                res.append((transcriptid[i], sentenceid[i], torch.mean(output[i,:end], 0)))
            
        # save every chunk
        torch.save(res, f'{save_path}_{start}.pt')    
        
        return res
        
                
PREENCODE_BATCH_SIZE = 128
MAX_SENT_LEN = 256

text_df = text_present_sentencized
start = 0
stop = 500 # len(text_df)
chunksize = 400000 # 400000 for 1/10 to tatal 

res = []
for i in range(start, stop, chunksize):
    print(f'Processing {i}/{stop}...{i/stop*100: .1f}% {Now()}')
    
    try:
        df = text_df.iloc[i:min(i+chunksize, stop)]
        if min(i+chunksize, stop) % 2 != 0:
            df = df.iloc[:-1]
        ds = CCDataset(df, transform=Tokenize(gpt_tokenizer, pad_token_id, max_seq_len=MAX_SENT_LEN))
        dl = DataLoader(ds, batch_size=PREENCODE_BATCH_SIZE,
                        shuffle=False, num_workers=0,
                        collate_fn=collate_fn,
                        drop_last=False,
                        pin_memory=False)

        res.extend(pre_encode(dl, model=gpt_model, save_path='./data/text_present_test', start=i))
    except Exception as e:
        print(f'Exception i={i}')
        print(f'   {e}')

# %% [markdown] toc-hr-collapsed=true toc-nb-collapsed=true
# ## SBERT

# %% [markdown]
# ### sentencize

# %% [markdown]
# Load documents

# %%
# load (X,Y)
# %time targets_df = pd.read_feather(f'{DATA_DIR}/f_sue_keydevid_car_finratio_transcriptid_text.feather')
print(f'num of calls: {len(targets_df)}')

# %% [markdown]
# Sentencize documents

# %%
# spacy model
nlp = English()  
sentencizer = nlp.create_pipe("sentencizer")
nlp.add_pipe(sentencizer)

def sentencize(df:str, sentencizer, text_field):
    transcriptids = df['transcriptid']
    texts = df[text_field]
    
    res = []
    texts_sentencized = nlp.pipe(texts, batch_size=1000, n_process=1)
    for tid, text in tqdm(zip(transcriptids, texts_sentencized), total=len(texts)):
        for sid, sent in enumerate(text.sents):
            res.append((tid, sid, sent.text))
        
    return res

text_all_sentencized = sentencize(targets_df, nlp, 'text_all')
text_present_sentencized = sentencize(targets_df, nlp, 'text_present')
text_qa_sentencized = sentencize(targets_df, nlp, 'text_qa')

# %% [markdown]
# Save results

# %%
pd.DataFrame(text_present_sentencized, columns=['transcriptid', 'sentenceid', 'text']).to_feather(f'{DATA_DIR}/text_present_sentencized.feather')

pd.DataFrame(text_qa_sentencized, columns=['transcriptid', 'sentenceid', 'text']).to_feather(f'{DATA_DIR}/text_qa_sentencized.feather')

pd.DataFrame(text_all_sentencized, columns=['transcriptid', 'sentenceid', 'text']).to_feather(f'{DATA_DIR}/text_all_sentencized.feather')

# %% [markdown]
# ### load model

# %%
model_path = "C:/Users/rossz/.cache/torch/sentence_transformers/public.ukp.informatik.tu-darmstadt.de_reimers_sentence-transformers_v0.2_roberta-large-nli-stsb-mean-tokens.zip"

with open(os.path.join(model_path, 'modules.json')) as fIn:
    contained_modules = json.load(fIn)
    
sbert_modules = OrderedDict()
for module_config in contained_modules:
    module_class = sentence_transformers.util.import_from_string(module_config['type'])
    module = module_class.load(os.path.join(model_path, module_config['path']))
    sbert_modules[module_config['name']] = module
    
sbert_pad_token_id = 1

sbert_model = nn.Sequential(sbert_modules)
sbert_model = nn.DataParallel(sbert_model)
sbert_model.to(cuda0);
log_gpu_memory();


# %% [markdown]
# ### define Dataset

# %%
class Tokenize():
    def __init__(self, modules, pad_token_id, max_seq_len):
        '''
        max_seq_len: There're still ass-cover statement in the call, which are very long.
            I remove every sentence which are longer than `max_seq_len`
        pad_token_id: for empty sentences, set length to 1 and fill with `pad_token_id`
        '''
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        self.modules = modules
        
    def __call__(self, sample):
        transcriptid, sentenceid, sent = sample
        sent = self.modules[next(iter(self.modules))].tokenize(sent)
        
        if len(sent) == 0 or len(sent) < self.max_seq_len:
            return transcriptid, sentenceid, sent
        else:
            return transcriptid, sentenceid, [self.pad_token_id]        


class CCDataset(Dataset):
    def __init__(self, df, transform=None):
        '''
        Args:
            df: DataFrame 
        '''
        self.transform = transform
        self.df = df
        self.length_sorted_idx = np.argsort([len(sent) for sent in df['text'].tolist()])

        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        # sample: (transcripid, sentenceid, text)
        sample = tuple(self.df.iloc[self.length_sorted_idx[idx]])
            
        if self.transform:
            sample = self.transform(sample)
            
        return sample
    
# MAX_SENT_LEN = 256
# ds = CCDataset(text_present_sentencized, transform=Tokenize(sbert_tokenizer, modules, pad_token_id=0, max_seq_len=MAX_SENT_LEN))

# %% [markdown]
# ### define DataLoader

# %%
# --------------------------- Create DataLoader--------------------------
def collate_fn(data: list, modules):
    '''
    Returns:
        featurs: a list of features. {'input_ids', 'input_mask', 'sentence_lengths'}
    '''
    transcriptids, sentenceids, sents = list(zip(*data))
    meta = (transcriptids, sentenceids)

    # valid seq_len
    valid_seq_len = [len(sent) for sent in sents]
    longest_seq_len = max(valid_seq_len)
    
    # pad
    features = {}
    for sent in sents:
        sentence_features = modules[next(iter(modules))].get_sentence_features(sent, longest_seq_len)
        
        for feature_name in sentence_features:
            if feature_name not in features:
                features[feature_name] = []
            features[feature_name].append(sentence_features[feature_name])
            
    for feature_name in features:
        features[feature_name] = torch.tensor(np.asarray(features[feature_name]))
            
    return {'features': features, 'meta': meta}

# dl = DataLoader(ds, batch_size=32,
#                 shuffle=False, num_workers=0,
#                 collate_fn=partial(collate_fn, modules=modules),
#                 drop_last=False,
#                 pin_memory=False)
# one_batch = next(iter(dl))
# one_batch


# %% [markdown]
# ### encode

# %%
def pre_encode_sbert(dl, model, save_path, start):
    with torch.no_grad():
        res = []
        for batch in tqdm(dl):
            features = batch['features']
            transcriptids, sentenceids = batch['meta']
            embeddings = model(features)['sentence_embedding'].to(cpu).numpy()
            
            for transcriptid, sentenceid, embedding in zip(transcriptids, sentenceids, embeddings):
                res.append((transcriptid, sentenceid, embedding))
            
        # save every chunk
        torch.save(res, f'{save_path}_{start}.pt')   
        
    return res


text_df = pd.read_feather(f'{DATA_DIR}/text_present_sentencized.feather')
start = 800000
stop = len(text_df)
chunksize = 400000 # 400000 for 1/10 to tatal 
MAX_SENT_LEN = 256
PREENCODE_BATCH_SIZE = 512

res = []
for i in range(start, stop, chunksize):
    print(f'Processing {i}/{stop}...{i/stop*100: .1f}% {Now()}')
    
    try:
        df = text_df.iloc[i:min(i+chunksize, stop)]
        if min(i+chunksize, stop) % 2 != 0:
            df = df.iloc[:-1]

        ds = CCDataset(df, transform=Tokenize(sbert_modules, pad_token_id=sbert_pad_token_id, max_seq_len=MAX_SENT_LEN))
        dl = DataLoader(ds, batch_size=PREENCODE_BATCH_SIZE,
                        shuffle=False, num_workers=0,
                        collate_fn=partial(collate_fn, modules=sbert_modules),
                        drop_last=False,
                        pin_memory=True)

        res.extend(pre_encode_sbert(dl, model=sbert_model, save_path='./data/text_present_sbert_roberta_nlistsb_encoded', start=i))
    except Exception as e:
        print(f'Exception i={i}')
        print(f'   {e}')

# %% [markdown]
# ## XLNet

# %% [markdown]
# ### load XLNet

# %%
# GPT-2
xlnet_tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')

xlnet_model = XLNetModel.from_pretrained('xlnet-base-cased')
xlnet_model = nn.DataParallel(xlnet_model)
xlnet_model.to(cuda0);

# %%
xlnet_tokenizer.vocab_size

# %% [markdown]
# ### test `seq_len` limits

# %%
# load (X,Y)
df = pd.read_feather(f'{DATA_DIR}/f_sue_keydevid_car_finratio_transcriptid_text.feather')
print(f'num of calls: {len(df)}')
df.iloc[:1]

# %%
text_present = [xlnet_tokenizer.encode(text) for text in df.text_present]
text_qa = [xlnet_tokenizer.encode(text) for text in df.text_qa]

# %%
seq_len_present = [len(text) for text in text_present]
seq_len_qa = [len(text) for text in text_qa]
seq_len_all = [x+y for x, y in zip(seq_len_present, seq_len_qa)]

# %%
import plotly.express as px
import plotly.graph_objects as go

fig = (go.Figure(go.Histogram(x=seq_len_present, nbinsx=100, name='present'))
    .add_histogram(x=seq_len_qa, name='qa')
    .add_histogram(x=seq_len_all, name='all')
    .update_layout(autosize=False))
fig.write_html('data/seq_len_tokenized_by_XLNet.html', include_plotlyjs='cdn')
fig

# %%
batch_size = 2
seq_len = 10000
vocab_size = 32000

with torch.no_grad():
    for i in tqdm(range(5)):
        inputs = torch.tensor(list(torch.utils.data.RandomSampler(range(vocab_size), replacement=True, num_samples=batch_size*seq_len))).reshape(batch_size, seq_len)
        y = xlnet_model(inputs)[0].to(cpu)


# %% [markdown] toc-hr-collapsed=true toc-nb-collapsed=true toc-hr-collapsed=true toc-nb-collapsed=true toc-hr-collapsed=true toc-nb-collapsed=true toc-hr-collapsed=true toc-nb-collapsed=true toc-hr-collapsed=true toc-nb-collapsed=true toc-hr-collapsed=true toc-nb-collapsed=true toc-hr-collapsed=true toc-nb-collapsed=true toc-hr-collapsed=true toc-nb-collapsed=true toc-hr-collapsed=true toc-nb-collapsed=true toc-hr-collapsed=true toc-nb-collapsed=true toc-hr-collapsed=true toc-nb-collapsed=true toc-hr-collapsed=true toc-nb-collapsed=true toc-hr-collapsed=true toc-nb-collapsed=true toc-hr-collapsed=true toc-nb-collapsed=true
# # Merge embeddings

# %% [markdown] toc-hr-collapsed=true toc-nb-collapsed=true
# ## merge pre-embedding

# %%
# %%time
def merge_preembeddings(preembedding_type, text_type):
    # load text_sentencied, (tid, sid, text)
    # which is used for checking embedding number
    if f'text_{text_type}_sentencized' not in globals():
        text_sentencized = pd.read_feather(f'{DATA_DIR}/text_{text_type}_sentencized.feather')
    
    # load preembeddings
    embedding_paths = [file for file in os.listdir(f'{DATA_DIR}/embeddings') if re.search(f'text_{preembedding_type}', file)]
    for path in embedding_paths: print(path)
        
    print(f'Loading preembeddings...')
    preembeddings_tmp = []
    for embedding_path in tqdm(embedding_paths):
        preembeddings_tmp.extend(torch.load(f'{DATA_DIR}/embeddings/{embedding_path}'))
        
    emb_dim = preembeddings_tmp[0][2].shape[0]
        
    # check if every sentence in `text_sentencized` has been preencoded
    # for every missing sentences, replacing with torch.zeros(1024)
    # The reason that some sentences have not been encoded is that the batch_size is 
    # an even number while the total number of sentences may be odd, in which case
    # the last sentence of the dataset will be removed.
    tid_sid_from_text_sentencized = set(f'{tid}-{sid}' for tid, sid in zip(text_sentencized.transcriptid, text_sentencized.sentenceid))

    tid_sid_from_preembeddings = set(f'{tid}-{sid}' for tid, sid, _ in preembeddings_tmp)

    for tid_sid in tid_sid_from_text_sentencized:
        if tid_sid not in tid_sid_from_preembeddings:
            tid, sid = tid_sid.split('-')
            tid = int(tid)
            sid = int(sid)
            text = text_sentencized.loc[(text_sentencized.transcriptid==tid) & (text_sentencized.sentenceid==sid)]['text'].values[0]
            print('Not found:')
            print(f'  trascriptid: {tid}  sentenceid: {sid}')
            print(f'  text: {text}')
            print(f'-----------')

            preembeddings_tmp.append((tid, sid, np.zeros(emb_dim)))
            
    assert len(preembeddings_tmp)==len(text_sentencized), 'preembedding # != sentence #'
        
    # sort by (transcriptid, sentenceid)
    print(f'sorting by (transcriptid, sentenceid)')
    preembeddings_tmp.sort(key=itemgetter(0,1))
    
    # group by transcriptid
    preembeddings_bytid = defaultdict(list)
    for transcriptid, _, emb in preembeddings_tmp:
        preembeddings_bytid[transcriptid].append(emb)

    preembeddings_bytid_stacked = {}
    print('Stacking embeddings...')
    for k, v in tqdm(preembeddings_bytid.items()):
        preembeddings_bytid_stacked[k] = torch.tensor(np.array(v))
    print(f'N call event: {len(preembeddings_bytid_stacked)}')

    return preembeddings_bytid_stacked



for text_type in ['all']:
    # merge preembeddings
    preembedding_type = f'{text_type}_sbert_roberta_nlistsb_encoded'
    preembedding_name = f'preembeddings_{preembedding_type}'
    preembeddings = merge_preembeddings(preembedding_type, text_type)
    # save preembeddings
    print(f'saving preembeddings...')
    torch.save(preembeddings, f'{DATA_DIR}/embeddings/{preembedding_name}.pt')

# %% [markdown] toc-hr-collapsed=true toc-nb-collapsed=true
# ## check `id-text` pair 

# %% [markdown]
# > Task: final check that id-text are correctly matched
# >
# > Check **Pass!**

# %%
cpu_model = SentenceTransformer('roberta-large-nli-stsb-mean-tokens')

# %%
text_sentencized = pd.read_feather(f'{DATA_DIR}/text_all_sentencized.feather')
targets_df = pd.read_feather(f'{DATA_DIR}/f_sue_keydevid_car_finratio_transcriptid_text.feather')

# %%
preembeddings[1441428][123]

# %%
text = text_sentencized[(text_sentencized.transcriptid==1441428) & (text_sentencized.sentenceid==123)]
text = text['text'].tolist()

cpu_model.encode(text)


# %%
class CCDataset(Dataset):
    
    def __init__(self, preembeddings: list, targets_df, split_window, split_type, transcriptids=None, transform=None):
        '''
        Args:
            preembeddings: list of embeddings. Each element is a tensor (S, E) where S is number of sentences in a call
            targets_df: DataFrame of targets variables.
            split_window: str. e.g., "roll-09"
            split_type: str. 'train' or 'test'
            transcriptids: list. If provided, only the given transcripts will be used in generating the Dataset. `transcriptids` is applied **on top of** `split_window` and `split_type`
        '''

        # get split dates from `split_df`
        _, train_start, train_end, test_start, test_end = tuple(split_df.loc[split_df.window==split_window].iloc[0])
        train_start = datetime.strptime(train_start, '%Y-%m-%d').date()
        train_end = datetime.strptime(train_end, '%Y-%m-%d').date()
        test_start = datetime.strptime(test_start, '%Y-%m-%d').date()
        test_end = datetime.strptime(test_end, '%Y-%m-%d').date()
        
        # select valid transcriptids (preemb_keys) according to split dates 
        if split_type=='train':
            transcriptids = targets_df[targets_df.ciq_call_date.between(train_start, train_end)].transcriptid.tolist()
        elif split_type=='test':
            transcriptids = targets_df[targets_df.ciq_call_date.between(test_start, test_end)].transcriptid.tolist()

        self.valid_preemb_keys = set(transcriptids).intersection(set(preembeddings.keys()))
        
        if transcriptids is not None:
            self.valid_preemb_keys = self.valid_preemb_keys.intersection(set(transcriptids))
        
        # self attributes
        self.targets_df = targets_df
        self.preembeddings = preembeddings
        self.transform = transform
        self.sent_len = sorted([(k, preembeddings[k].shape[0]) 
            for k in self.valid_preemb_keys],
            key=itemgetter(1))
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
        
        # inputs: preembeddings
        embeddings = self.preembeddings[transcriptid]
        
        # all of the following targests are
        # of type `numpy.float64`
        mcap = targets.mcap
        sue = targets.sue
        suelag = targets.sue_lag1
        selag = targets.se_lag1
        se = targets.se
        selead = targets.se_lead1
        sestlag = targets.sest_lag1
        sest = targets.sest
        car_0_30 = targets.car_0_30
        car_0_30_lag = targets.car_0_30_lag1
        
        return transcriptid, embeddings, mcap, suelag, sue, car_0_30_lag, car_0_30, se, selead, sestlag, sest


# # test DataSet...
# targets_df_path = f'{DATA_DIR}/f_sue_keydevid_car_finratio_transcriptid_text.feather'
# preembedding_type = 'ques_sbert_roberta_nlistsb_encoded'

# # load preembeddings
# if 'preembeddings' not in globals():
#     print(f'Loading preembeddings...{Now()}')
#     preembeddings = torch.load(f'{DATA_DIR}/embeddings/preembeddings_{preembedding_type}.pt')
#     print(f'Loading finished. {Now()}')
    
# # load targets_df
# if 'targets_df' not in globals():
#     targets_df = pd.read_feather(targets_df_path)

# # choose train/val split
# split_df = pd.read_csv(f'{DATA_DIR}/split_dates.csv')

# # create Dataset
# test_ds = CCDataset(preembeddings, targets_df, split_window='roll-19', split_type='train')

# test_ds[876]

# %% [markdown]
# ## compute # rolling samples

# %%
# %%time
# load `targets_df`, `split_df`, `preembeddings`
if 'targets_df' not in globals():
    targets_df = pd.read_feather(f'{DATA_DIR}/f_sue_keydevid_car_finratio_transcriptid_text.feather')

split_df = pd.read_csv(f'{DATA_DIR}/split_dates.csv')

if 'preembeddings' not in globals():
    preembeddings = torch.load(f'{DATA_DIR}/embeddings/preembeddings_qa_sbert_roberta_nlistsb_encoded.pt')

# %%
split_windows = split_df.window.tolist()

n_train_samples, n_test_samples = [], []

for split_window in split_windows:
    print(f'{split_window}:')
    for split_type in ['train', 'test']:
        ds = CCDataset(preembeddings, targets_df, split_window, split_type)
        if split_type=='train':
            print(f'    {split_type}:{ds.n_samples} ({ds.train_start}, {ds.train_end})')
            n_train_samples.append(ds.n_samples)
        elif split_type=='test':
            print(f'    {split_type}:{ds.n_samples} ({ds.test_start}, {ds.test_end})')    
            n_test_samples.append(ds.n_samples)

# %%
# statistics
np.mean(n_train_samples)
np.mean(n_test_samples)


# %% [markdown]
# # Base

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
def refresh_ckpt(ckpt_path):
    '''
    move all `.ckpt` files to `/temp`
    '''
    for name in os.listdir(ckpt_path):
        if name.endswith('.ckpt'):
            shutil.move(f'{ckpt_path}/{name}', f'{ckpt_path}/temp/{name}')

# helpers: load targets
def load_targets(targets_name):
    if 'targets_df' not in globals():
        globals()['targets_df'] = pd.read_feather(f'{DATA_DIR}/{targets_name}.feather')
        
# helpers: load preembeddings
def load_preembeddings(preembedding_type):
    if 'preembeddings' not in globals():
        print(f'Loading preembeddings...{Now()}')
        globals()['preembeddings'] = torch.load(f"{DATA_DIR}/embeddings/preembeddings_{preembedding_type}.pt")
        print(f'Loading finished. {Now()}')
        
# helpers: load split_df
def load_split_df(roll_type):
    split_df = pd.read_csv(f'{DATA_DIR}/split_dates.csv')
    globals()['split_df'] = split_df.loc[split_df.roll_type==roll_type]


# %%
# loop one
def train_one(Model, window_i, model_hparams, train_hparams):
    global split_df, targets_df
    
    # set window
    model_hparams.update({'window': split_df.iloc[window_i].window})
    
    # init model
    model = Model(Namespace(**model_hparams))

    # get model type
    train_hparams['task_type'] = model.task_type
    train_hparams['feature_type'] = model.feature_type
    train_hparams['model_type'] = model.model_type
    train_hparams['attn_type'] = model.attn_type

    # checkpoint
    ckpt_prefix = f"{train_hparams['model_type']}_{model_hparams['window']}_"
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        verbose=True,
        mode='min',
        monitor='val_loss',
        prefix=ckpt_prefix,
        filepath=train_hparams['checkpoint_path'],
        save_top_k=train_hparams['save_top_k'],
        period=train_hparams['checkpoint_period'])

    # logger
    logger = pl.loggers.CometLogger(
        api_key=os.environ.get('COMET_API_KEY'),
        save_dir='/data/logs',
        project_name='earnings-call',
        experiment_name=model_hparams['window'],
        workspace='amiao',
        display_summary_level=0)

    # early stop
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=train_hparams['early_stop_patience'],
        verbose=True,
        mode='min')

    # trainer
    trainer = pl.Trainer(default_root_dir=train_hparams['checkpoint_path'], 
                         checkpoint_callback=checkpoint_callback, 
                         early_stop_callback=early_stop_callback,
                         overfit_pct=train_hparams['overfit_pct'], 
                         row_log_interval=train_hparams['row_log_interval'],
                         val_check_interval=train_hparams['val_check_interval'], 
                         progress_bar_refresh_rate=2, 
                         gpus=-1, 
                         distributed_backend='dp', 
                         accumulate_grad_batches=train_hparams['accumulate_grad_batches'],
                         min_epochs=train_hparams['min_epochs'],
                         max_epochs=train_hparams['max_epochs'], 
                         max_steps=train_hparams['max_steps'], 
                         logger=logger, 
                         close_after_fit=True)

    # delete unused hparam
    if model.model_type=='mlp': model_hparams.pop('final_tdim',None)
    if model.feature_type=='fin-ratio': 
        model_hparams.pop('preembedding_type',None)
        model_hparams.pop('max_seq_len',None)
        model_hparams.pop('n_layers_encoder',None)
        model_hparams.pop('n_head_encoder',None)
        model_hparams.pop('d_model',None)
        model_hparams.pop('dff',None)
    if model.feature_type!='text + fin-ratio': 
        model_hparams.pop('normalize_layer',None)
        model_hparams.pop('normalize_batch',None)
    if model.attn_type!='mha': model_hparams.pop('n_head_decoder',None)

    # add n_model_params
    train_hparams['n_model_params'] = sum(p.numel() for p in model.parameters())

    # upload hparams
    logger.experiment.log_parameters(model_hparams)
    logger.experiment.log_parameters(train_hparams)

    # refresh GPU memory
    refresh_cuda_memory()

    # fit and test
    try:
        # train the model
        trainer.fit(model)

        # load back the best model 
        best_model_name = sorted([f"{train_hparams['checkpoint_path']}/{model_name}" 
                                  for model_name in os.listdir(train_hparams['checkpoint_path']) 
                                  if model_name.startswith(ckpt_prefix)])[-1]
        print(f'loading best model: {best_model_name}')
        best_model = Model.load_from_checkpoint(best_model_name)
        best_model.freeze()

        # test on the best model
        trainer.test(best_model, test_dataloaders=model.test_dataloader())

    except RuntimeError as e:
        raise e
    finally:
        del model, trainer
        refresh_cuda_memory()
        logger.finalize('finished')


# %%
# Dataset: Txt + Fin-ratio
class CCDataset(Dataset):
    
    def __init__(self, split_window, split_type, text_in_dataset, roll_type, print_window, transcriptids=None, transform=None):
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

        self.text_in_dataset = text_in_dataset
        
        # decalre data as globals so don't need to create/reload
        global preembeddings, targets_df, split_df
        
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
            transcriptids = targets_df[targets_df.ciq_call_date.between(train_start, train_end)].transcriptid.tolist()
        elif split_type=='test':
            transcriptids = targets_df[targets_df.ciq_call_date.between(test_start, test_end)].transcriptid.tolist()

        self.valid_preemb_keys = set(transcriptids).intersection(set(preembeddings.keys()))
        
        if transcriptids is not None:
            self.valid_preemb_keys = self.valid_preemb_keys.intersection(set(transcriptids))
        
        # self attributes
        self.targets_df = targets_df
        self.preembeddings = preembeddings
        self.transform = transform
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
        
        # inputs: preembeddings
        embeddings = self.preembeddings[transcriptid]
        
        # all of the following targests are
        # of type `numpy.float64`
        sue = targets.sue
        sest = targets.sest
        car_0_30 = targets.car_0_30
        
        alpha = targets.alpha
        volatility = targets.volatility
        mcap = targets.mcap/1e6
        bm = targets.bm
        roa = targets.roa
        debt_asset = targets.debt_asset
        numest = targets.numest
        smedest = targets.smedest
        sstdest = targets.sstdest
        car_m1_m1 = targets.car_m1_m1
        car_m2_m2 = targets.car_m2_m2
        car_m30_m3 = targets.car_m30_m3
        volume = targets.volume
        
        if self.text_in_dataset:
            return car_0_30, transcriptid, embeddings, alpha, car_m1_m1, car_m2_m2, car_m30_m3, \
                   sest, sue, numest, sstdest, smedest, \
                   mcap, roa, bm, debt_asset, volatility, volume
        else:
            return torch.tensor(car_0_30,dtype=torch.float32), \
                   torch.tensor([alpha, car_m1_m1, car_m2_m2, car_m30_m3, sest, sue, numest, sstdest, smedest, mcap, roa, bm, debt_asset, volatility, volume], dtype=torch.float32)
    
# Model: position encoder
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
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
        
        self.hparams = hparams
        # self.text_in_dataset will be filled during instanciating.

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
                                       roll_type=self.hparams.roll_type, print_window=True)
        self.val_dataset = CCDataset(self.hparams.window, split_type='test', text_in_dataset=self.text_in_dataset,
                                     roll_type=self.hparams.roll_type, print_window=False)
        self.test_dataset = CCDataset(self.hparams.window, split_type='test', text_in_dataset=self.text_in_dataset, 
                                      roll_type=self.hparams.roll_type, print_window=False)

    # DataLoader
    def train_dataloader(self):
        '''
        Caution:
        - If you enable `BatchNorm`, then must set `drop_last=True`.
        '''
        collate_fn = self.collate_fn if self.text_in_dataset else None
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True, drop_last=True, num_workers=0, pin_memory=True, collate_fn=collate_fn)
    
    def val_dataloader(self):
        '''
        Caution: 
        - To improve the validation speed, I'll set val_batch_size to 4. 
        - Must set `drop_last=True`, otherwise the `val_loss` tensors for different batches won't match and hence give you error.
        - Not to set `val_batch_size` too large (e.g., 16), otherwise you'll lose precious validation data points
        '''
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
        car_0_30, transcriptid, embeddings, alpha, car_m1_m1, car_m2_m2, car_m30_m3, \
        sest, sue, numest, sstdest, smedest, \
        mcap, roa, bm, debt_asset, volatility, volume = zip(*data)
            
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

        return torch.tensor(car_0_30, dtype=torch.float32), torch.tensor(transcriptid, dtype=torch.float32), \
               embeddings.float(), mask, \
               torch.tensor(alpha, dtype=torch.float32), torch.tensor(car_m1_m1, dtype=torch.float32), \
               torch.tensor(car_m2_m2, dtype=torch.float32), torch.tensor(car_m30_m3, dtype=torch.float32), \
               torch.tensor(sest, dtype=torch.float32), torch.tensor(sue, dtype=torch.float32), \
               torch.tensor(numest, dtype=torch.float32), torch.tensor(sstdest, dtype=torch.float32), \
               torch.tensor(smedest, dtype=torch.float32), torch.tensor(mcap, dtype=torch.float32), \
               torch.tensor(roa, dtype=torch.float32), torch.tensor(bm, dtype=torch.float32), \
               torch.tensor(debt_asset, dtype=torch.float32), torch.tensor(volatility, dtype=torch.float32), \
               torch.tensor(volume, dtype=torch.float32)
        
    # optimizer
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer   


# %% [markdown]
# # MLP

# %%
# FC
class CCMLP(CC):
    def __init__(self, hparams):
        super().__init__(hparams)
        
        self.hparams = hparams
        
        # attibutes
        self.task_type = 'single'
        self.feature_type = 'fin-ratio'
        self.model_type = 'mlp'
        self.attn_type = 'dotprod'
        
        self.text_in_dataset = True if self.feature_type!='fin-ratio' else False 
        
        # dropout layers
        self.dropout1 = nn.Dropout(hparams.dropout)
        self.dropout2 = nn.Dropout(hparams.dropout)
        
        # fc layers
        self.linear1 = nn.Linear(14, 64)
        self.linear2 = nn.Linear(64, 64)
        self.linear3 = nn.Linear(64, 1)

    # forward
    def forward(self, inp):
        x_car = self.dropout1(F.relu(self.linear1(inp)))
        x_car = self.dropout2(F.relu(self.linear2(x_car)))
        y_car = self.linear3(x_car) # (N, 1)
        
        return y_car
    
        
    # train step
    def training_step(self, batch, idx):
        
        car, inp = batch
        
        # forward
        y_car = self.forward(inp) # (N, 1)

        # compute loss
        loss_car = self.mse_loss(y_car, car.unsqueeze(-1)) # ()
        
        # logging
        return {'loss': loss_car, 'log': {'train_loss': loss_car}}
            
    # validation step
    def validation_step(self, batch, idx):
        
        car, inp = batch
        
        # forward
        y_car = self.forward(inp) # (N, 1)

        # compute loss
        loss_car = self.mse_loss(y_car, car.unsqueeze(-1)) # ()
        
        # logging
        return {'val_loss': loss_car}        
        
    # test step
    def test_step(self, batch, idx):
        
        car, inp = batch
        
        # forward
        y_car = self.forward(inp) # (N, 1)

        # compute loss
        loss_car = self.mse_loss(y_car, car.unsqueeze(-1)) # ()
        
        # logging
        return {'test_loss': loss_car}  

# %%
# loop over 24 windows
load_split_df()
load_targets()

# model hparams
model_hparams = {
    'preembedding_type': 'present_sbert_roberta_nlistsb_encoded',
    'batch_size': 8192,
    'val_batch_size': 1,
    'learning_rate': 3e-4,
    'task_weight': 1,
    'dropout': 0.5} # optional

# train hparams
train_hparams = {
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
refresh_ckpt(train_hparams['checkpoint_path'])
    
# loop over 24!
for window_i in range(len(split_df))[:1]:
    # load preembeddings
    load_preembeddings(model_hparams['preembedding_type'])
    
    # train one window
    train_one(CCMLP, window_i, model_hparams, train_hparams)


# %% [markdown] toc-hr-collapsed=true toc-nb-collapsed=true toc-hr-collapsed=true toc-nb-collapsed=true toc-hr-collapsed=true toc-nb-collapsed=true toc-hr-collapsed=true toc-nb-collapsed=true toc-hr-collapsed=true toc-nb-collapsed=true
# # RNN

# %% [markdown]
# ## Model

# %%
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
        return {'loss': loss_car, 'log': {'train_loss': loss_car}}
            
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


# %% [markdown] toc-hr-collapsed=true toc-nb-collapsed=true
# ## run

# %%
# loop over 24 windows
load_split_df()
load_targets()

# model hparams
model_hparams = {
    'preembedding_type': 'all_sbert_roberta_nlistsb_encoded',
    'batch_size': 8,
    'val_batch_size': 1,
    'max_seq_len': 1024, # optional
    'learning_rate': 3e-4,
    'task_weight': 1,
    'text_in_dataset': True,

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
train_hparams = {
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
refresh_ckpt(train_hparams['checkpoint_path'])
    
# loop over 24!
for window_i in range(len(split_df)):
    # load preembeddings
    load_preembeddings(model_hparams['preembedding_type'])

    # train one window
    train_one(CCGRU, window_i, model_hparams, train_hparams)


# %% [markdown]
# # Transformer

# %% [markdown]
# ## STL

# %%
# STL-text-MLP
class CCTransformerSTLTxt(CC):
        
    def __init__(self, hparams):
        super().__init__(hparams)
        
        self.hparams = hparams
        
        # specify model type
        self.task_type = 'single'
        self.feature_type = 'text'
        self.attn_type = 'dotprod'
        self.model_type = 'transformer'
        self.text_in_dataset = True if self.feature_type!='fin-ratio' else False 
        
        # positional encoding
        self.encoder_pos = PositionalEncoding(hparams.d_model, hparams.attn_dropout)
        
        # encoder layers for input, expert, nonexpert
        encoder_layers_expert = nn.TransformerEncoderLayer(hparams.d_model, hparams.n_head_encoder, hparams.dff, hparams.attn_dropout)
        
        # atten layers for SUE, CAR, SELEAD, SEST
        self.attn_layers_car = nn.Linear(hparams.d_model, 1)
        self.attn_dropout = nn.Dropout(hparams.attn_dropout)
        
        # Build Encoder and Decoder
        self.encoder_expert = nn.TransformerEncoder(encoder_layers_expert, hparams.n_layers_encoder)
        
        # linear layer to produce final result
        self.linear_car_1 = nn.Linear(hparams.d_model, hparams.d_model)
        self.linear_car_2 = nn.Linear(hparams.d_model, hparams.final_tdim)
        self.linear_car_3 = nn.Linear(hparams.final_tdim, 1)
        
        self.dropout_1 = nn.Dropout(hparams.dropout)
        self.dropout_2 = nn.Dropout(hparams.dropout)
        
    # forward
    def forward(self, inp, src_key_padding_mask):
        bsz, embed_dim = inp.size(0), inp.size(2)
        
        # if S is longer than max_seq_len, cut
        inp = inp[:,:self.hparams.max_seq_len,] # (N, S, E)
        src_key_padding_mask = src_key_padding_mask[:,:self.hparams.max_seq_len] # (N, S)
        
        inp = inp.transpose(0, 1) # (S, N, E)
        
        # positional encoding
        x = self.encoder_pos(inp) # (S, N, E)
        
        # encode
        x_expert = self.encoder_expert(x, src_key_padding_mask=src_key_padding_mask).transpose(0,1) # (N, S, E)
        
        # decode with attn
        x_attn = self.attn_dropout(F.softmax(self.attn_layers_car(x_expert), dim=1)) # (N, S, 1)
        y_car = torch.bmm(x_expert.transpose(-1,-2), x_attn).squeeze(-1) # (N, E)
        
        # final linear layer
        y_car = self.dropout_1(F.relu(self.linear_car_1(y_car)))
        y_car = self.dropout_2(F.relu(self.linear_car_2(y_car)))
        y_car = self.linear_car_3(y_car) # (N,1)
        
        # final output
        return y_car
    
    # traning step
    def training_step(self, batch, idx):
        
        car, transcriptid, embeddings, mask, alpha, car_m1_m1, car_m2_m2, car_m30_m3,\
        sest, sue, numest, sstdest, smedest, \
        mcap, roa, bm, debt_asset, volatility, volume = batch
        
        # get batch size
        bsz = sue.size(0)
        
        # forward
        y_car = self.forward(embeddings, mask) # (N, 1)

        # compute loss
        loss_car = self.mse_loss(y_car, car.unsqueeze(-1)).unsqueeze(-1) # (1,)
        
        # logging
        return {'loss': loss_car, 'log': {'train_loss': loss_car}}
        
    # validation step
    def validation_step(self, batch, idx):
        car, transcriptid, embeddings, mask, alpha, car_m1_m1, car_m2_m2, car_m30_m3,\
        sest, sue, numest, sstdest, smedest, \
        mcap, roa, bm, debt_asset, volatility, volume = batch
        
        # get batch size
        bsz = sue.size(0)

        # forward
        y_car = self.forward(embeddings, mask) # (N, 1)

        # compute loss
        loss_car = self.mse_loss(y_car, car.unsqueeze(-1)).unsqueeze(-1) # (1,)

        # logging
        return {'val_loss': loss_car}

    # test step
    def test_step(self, batch, idx):
        car, transcriptid, embeddings, mask, alpha, car_m1_m1, car_m2_m2, car_m30_m3,\
        sest, sue, numest, sstdest, smedest, \
        mcap, roa, bm, debt_asset, volatility, volume = batch
        
        # get batch size
        bsz = sue.size(0)

        # forward
        y_car = self.forward(embeddings, mask) # (N, 1)

        # compute loss
        loss_car = self.mse_loss(y_car, car.unsqueeze(-1)).unsqueeze(-1) # (1,)

        # logging
        return {'test_loss': loss_car}  


# %%
# STL-text-fr-MLP
class CCTransformerSTLTxtFr(CC):
    def __init__(self, hparams):
        super().__init__(hparams)
        
        self.hparams = hparams
        
        # specify model type
        self.task_type = 'single'
        self.feature_type = 'text + fin-ratio'
        self.attn_type = 'dotprod'
        self.model_type = 'transformer'
        self.text_in_dataset = True if self.feature_type!='fin-ratio' else False 

        self.n_covariate = 14
        
        # positional encoding
        self.encoder_pos = PositionalEncoding(hparams.d_model, hparams.attn_dropout)
        
        # encoder layers for input, expert, nonexpert
        encoder_layers_expert = nn.TransformerEncoderLayer(hparams.d_model, hparams.n_head_encoder, hparams.dff, hparams.attn_dropout)
        
        # atten layers for SUE, CAR, SELEAD, SEST
        self.attn_layers_car = nn.Linear(hparams.d_model, 1)
        self.attn_dropout_1 = nn.Dropout(hparams.attn_dropout)
        
        # Build Encoder and Decoder
        self.encoder_expert = nn.TransformerEncoder(encoder_layers_expert, hparams.n_layers_encoder)
        
        # linear layer to produce final result
        self.linear_car_1 = nn.Linear(hparams.d_model, hparams.d_model)
        self.linear_car_2 = nn.Linear(hparams.d_model, hparams.final_tdim)
        self.linear_car_3 = nn.Linear(hparams.final_tdim+self.n_covariate, hparams.final_tdim+self.n_covariate)
        self.linear_car_4 = nn.Linear(hparams.final_tdim+self.n_covariate, hparams.final_tdim+self.n_covariate)
        self.linear_car_5 = nn.Linear(hparams.final_tdim+self.n_covariate, 1)
        
        # dropout for final fc layers
        self.final_dropout_1 = nn.Dropout(hparams.dropout)
        self.final_dropout_2 = nn.Dropout(hparams.dropout)
        self.final_dropout_3 = nn.Dropout(hparams.dropout)
        
        # layer normalization
        if hparams.normalize_layer:
            self.layer_norm = nn.LayerNorm(hparams.final_tdim+self.n_covariate)
            
        # batch normalization
        if hparams.normalize_batch:
            self.batch_norm = nn.BatchNorm1d(hparams.final_tdim+self.n_covariate)

    # forward
    def forward(self, inp, src_key_padding_mask, alpha, car_m1_m1, car_m2_m2, car_m30_m3, sest, sue, numest, sstdest, \
                mcap, roa, bm, debt_asset, volatility, volume):
        bsz, embed_dim = inp.size(0), inp.size(2)
        
        # if S is longer than max_seq_len, cut
        inp = inp[:,:self.hparams.max_seq_len,] # (N, S, E)
        src_key_padding_mask = src_key_padding_mask[:,:self.hparams.max_seq_len] # (N, S)
        
        inp = inp.transpose(0, 1) # (S, N, E)
        
        # positional encoding
        x = self.encoder_pos(inp) # (S, N, E)
        
        # encode
        x_expert = self.encoder_expert(x, src_key_padding_mask=src_key_padding_mask).transpose(0,1) # (N, S, E)
        
        # decode with attn
        x_attn = self.attn_dropout_1(F.softmax(self.attn_layers_car(x_expert), dim=1)) # (N, S, 1)
        x_expert = torch.bmm(x_expert.transpose(-1,-2), x_attn).squeeze(-1) # (N, E)
        
        # mix with covariate
        x_expert = self.final_dropout_1(F.relu(self.linear_car_1(x_expert))) # (N, E)
        x_expert = F.relu(self.linear_car_2(x_expert)) # (N, final_tdim)
        x_final = torch.cat([x_expert, alpha.unsqueeze(-1), car_m1_m1.unsqueeze(-1), car_m2_m2.unsqueeze(-1),
                             car_m30_m3.unsqueeze(-1), sest.unsqueeze(-1), sue.unsqueeze(-1), numest.unsqueeze(-1),
                             sstdest.unsqueeze(-1), mcap.unsqueeze(-1), roa.unsqueeze(-1), bm.unsqueeze(-1),
                             debt_asset.unsqueeze(-1), volatility.unsqueeze(-1), volume.unsqueeze(-1)], dim=-1) # (N, X + final_tdim) where X is the number of covariate (n_covariate)
        
        # batch normalization
        if self.hparams.normalize_batch:
            x_final = self.batch_norm(x_final)
        
        # layer normalization
        if self.hparams.normalize_layer:
            x_final = self.layer_norm(x_final)
            
        # final FC
        x_final = self.final_dropout_2(F.relu(self.linear_car_3(x_final))) # (N, X + final_tdim)
        x_final = self.final_dropout_3(F.relu(self.linear_car_4(x_final))) # (N, X + final_tdim)
        y_car = self.linear_car_5(x_final) # (N,1)
        
        # final output
        return y_car
    
    # traning step
    def training_step(self, batch, idx):
        
        car, transcriptid, embeddings, mask, alpha, car_m1_m1, car_m2_m2, car_m30_m3,\
        sest, sue, numest, sstdest, smedest, \
        mcap, roa, bm, debt_asset, volatility, volume = batch
        
        # get batch size
        bsz = sue.size(0)
        
        # forward
        y_car = self.forward(embeddings, mask, alpha, car_m1_m1, car_m2_m2, car_m30_m3, sest, sue, numest, sstdest, \
                             mcap, roa, bm, debt_asset, volatility, volume) # (N, 1)

        # compute loss
        loss_car = self.mse_loss(y_car, car.unsqueeze(-1)).unsqueeze(-1) # (1,)
        
        # logging
        return {'loss': loss_car, 'log': {'train_loss': loss_car}}
        
    # validation step
    def validation_step(self, batch, idx):
        car, transcriptid, embeddings, mask, alpha, car_m1_m1, car_m2_m2, car_m30_m3,\
        sest, sue, numest, sstdest, smedest, \
        mcap, roa, bm, debt_asset, volatility, volume = batch
        
        # get batch size
        bsz = sue.size(0)

        # forward
        y_car = self.forward(embeddings, mask, alpha, car_m1_m1, car_m2_m2, car_m30_m3, sest, sue, numest, sstdest, \
                             mcap, roa, bm, debt_asset, volatility, volume) # (N, 1)

        # compute loss
        loss_car = self.mse_loss(y_car, car.unsqueeze(-1)).unsqueeze(-1) # (1,)

        # logging
        return {'val_loss': loss_car}

    # test step
    def test_step(self, batch, idx):
        car, transcriptid, embeddings, mask, alpha, car_m1_m1, car_m2_m2, car_m30_m3,\
        sest, sue, numest, sstdest, smedest, \
        mcap, roa, bm, debt_asset, volatility, volume = batch
        
        # get batch size
        bsz = sue.size(0)

        # forward
        y_car = self.forward(embeddings, mask, alpha, car_m1_m1, car_m2_m2, car_m30_m3, sest, sue, numest, sstdest, \
                             mcap, roa, bm, debt_asset, volatility, volume) # (N, 1)

        # compute loss
        loss_car = self.mse_loss(y_car, car.unsqueeze(-1)).unsqueeze(-1) # (1,)

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
    'targets_name': 'f_sue_keydevid_car_finratio_vol_transcriptid_sim_text', # key!
    'roll_type': '3y',  # key!
    'batch_size': 20,
    'val_batch_size': 4,
    'max_seq_len': 768, 
    'learning_rate': 2.5e-4,
    'task_weight': 1,
    'normalize_layer': False,
    'normalize_batch': True,

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
    'note': 'temp',
    'remove_outlier': 'no', # key!
    'checkpoint_path': 'D:\Checkpoints\earnings-call',
    'row_log_interval': 1,
    'save_top_k': 1,
    'val_check_interval': 0.2,

    # data size
    'overfit_pct': 1,
    'min_epochs': 3,
    'max_epochs': 10,
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
refresh_ckpt(train_hparams['checkpoint_path'])

# load split_df
load_split_df(model_hparams['roll_type'])
    
# load targets_df
load_targets(model_hparams['targets_name'])

# loop over 24!
for window_i in range(len(split_df))[:2]:
    # load preembeddings
    load_preembeddings(model_hparams['preembedding_type'])

    # train one window
    train_one(Model, window_i, model_hparams, train_hparams)


# %% [markdown]
# # Predict

# %%
# test on one batch
def test_step_text(model, batch):
    car, transcriptid, embeddings, mask, alpha, car_m1_m1, car_m2_m2, car_m30_m3,\
    sest, sue, numest, sstdest, smedest, \
    mcap, roa, bm, debt_asset, volatility, volume = batch

    # forward
    y_car = model(embeddings, mask).item() # (N, 1)
    transcriptid = transcriptid.int().item()
    docid = targets_df.loc[targets_df.transcriptid==transcriptid].docid.iloc[0]

    return y_car, docid, transcriptid

def test_step_text_fr(model, batch):
    car, transcriptid, embeddings, mask, alpha, car_m1_m1, car_m2_m2, car_m30_m3,\
    sest, sue, numest, sstdest, smedest, \
    mcap, roa, bm, debt_asset, volatility, volume = batch

    # forward
    y_car = model(embeddings, mask, alpha, car_m1_m1, car_m2_m2, car_m30_m3, sest, sue, numest, sstdest, \
                  mcap, roa, bm, debt_asset, volatility, volume).item() # (N, 1)
    transcriptid = transcriptid.int().item()
    docid = targets_df.loc[targets_df.transcriptid==transcriptid].docid.iloc[0]

    return y_car, docid, transcriptid

# test on one window
def test_one_window(model, dataloader):
    # select test_step
    if type(model) in [CCTransformerSTLTxt]:
        test_step = test_step_text
    elif type(model) in [CCTransformerSTLTxtFr]:
        test_step = test_step_text_fr
    
    ys = []
    for i, batch in enumerate(tqdm(dataloader)):
        ys.append(test_step(model, batch))
        # if i>=2: break
    return pd.DataFrame(ys, columns=['y_car', 'docid', 'transcriptid'])

# ground truth
def get_car_ranking_truth():
    car_ranking = pd.read_feather('data/car_ranking.feather')
    car_ranking = car_ranking.loc[car_ranking.roll_type==roll_type]
    return car_ranking

# get prediction
def get_car_ranking_predict():
    global hparams
    
    # load split_df
    load_split_df(hparams.roll_type)

    # load targets_df
    load_targets(hparams.targets_name)

    # load preembedding
    load_preembeddings(hparams.preembedding_type)
    
    car_ranking_predict = []
    for name in sorted(os.listdir(f'D:/Checkpoints/earnings-call/{hparams.ckpt_folder}')):
        if name.endswith('.ckpt'):

            # get window
            window = re.search(r'roll-\d{2}', name).group()

            # load model
            model = hparams.Model.load_from_checkpoint(f'D:/Checkpoints/earnings-call/{hparams.ckpt_folder}/{name}')

            # get testloader
            model.prepare_data()
            test_dataloader = model.test_dataloader()
            model.freeze()
            
            # predict
            y = test_one_window(model, test_dataloader)
            y['window'] = window
            y['roll_type'] = hparams.roll_type

            # append to ys
            car_ranking_predict.append(y)
    car_ranking_predict = pd.concat(car_ranking_predict)

    car_ranking_predict.reset_index().to_feather(f'data/{hparams.save_name}.feather')
    return car_ranking_predict


# %%
hparams = {
    'Model': CCTransformerSTLTxtFr,
    'ckpt_folder': '3y-TSFM-stl-text-fr',
    'save_name': 'car_ranking_3y_text_fr',

    'targets_name': 'f_sue_keydevid_car_finratio_vol_transcriptid_sim_text',
    'preembedding_type': 'all_sbert_roberta_nlistsb_encoded',
}
hparams = Namespace(**hparams)

# get roll_type
hparams.roll_type = hparams.ckpt_folder.split('/')[-1].split('-')[0]

# get car_ranking_predict
car_ranking_predict = get_car_ranking_predict()

# %% [markdown]
# # Comet Log

# %% [markdown]
# **Task:**
# - modify experiment attributes

# %%
query = ((comet_ml.api.Metric('test_rmse')!=None) &
         (comet_ml.api.Parameter('note')=='temp'))

exps = comet_ml.api.API().query('amiao', 'earnings-call', query, archived=False)

for exp in exps:
    exp.log_parameter('note', '3y-TSFM-stl-text-fr')

# %% [markdown]
# **Task:**
# - download comet log

# %%
# %%time
exps = comet_ml.api.API().query('amiao', 'earnings-call', comet_ml.api.Metric('test_rmse') != None, archived=False)

log_comet = []
for exp in exps:
    # get parameter
    log = {param['name']:param['valueCurrent'] for param in exp.get_parameters_summary()}
    
    # get metrics
    log['test_rmse'] = exp.get_metrics('test_rmse')[0]['metricValue']
    
    # get metadat
    log = {**log, **exp.get_metadata()}
    
    for key in ['checkpoint_path', 'f']:
        del log[key]
    log_comet.append(log)
    
log_comet = pd.DataFrame(log_comet)
log_comet.to_feather('data/comet_log.feather')
