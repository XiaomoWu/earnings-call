from jupyter_startup import *

# import tensorflow as tf
import comet_ml
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_memlab import LineProfiler
from collections import OrderedDict, defaultdict
from argparse import Namespace

from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data.dataset import random_split
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import spacy
from spacy.lang.en import English
from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER, HYPHENS
from spacy.lang.char_classes import CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS
from spacy.util import compile_infix_regex
from scipy.sparse import coo_matrix


# working directory
ROOT_DIR = 'C:/Users/rossz/OneDrive/CC'
DATA_DIR = f'{ROOT_DIR}/data'
print(f'ROOT_DIR: {ROOT_DIR}')
print(f'DATA_DIR: {DATA_DIR}')

# set random seed
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

# set device 'cuda' or 'cpu'
if torch.cuda.is_available():
    n_cuda = torch.cuda.device_count()
    
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
            
    print(f'\n{n_cuda} GPUs found:')
    for _ in range(n_cuda):
        globals()[f'cuda{_}'] = torch.device(f'cuda:{_}')
        print(f'    {torch.cuda.get_device_name(_)} (cuda{_})')
        
    print('\nGPU memory:')
    log_gpu_memory()
else:
    print('GPU NOT enabled')
    
cpu = torch.device('cpu')
n_cpu = int(mp.cpu_count()/2)

print(f'\nCPU count (physical): {n_cpu}')

ld('cc_ngram')

class CCCorpus():
    def __init__(self, ngram_type: str):
        '''
        ngram_type: 'unigram' or 'bigram'
        '''
        assert 'cc_ngram' in globals(), 'Load `cc_ngram` first!'
        global cc_ngram
        
        self.ngram_type = ngram_type
        self.docid = cc_ngram.docid.to_numpy()
        self.ndoc = len(self.docid)
        
        # create vocab
        self.get_vocab()
    def get_vocab(self):
        vocab = set(itertools.chain(*[d.keys() for d in cc_ngram[f'text_all_{self.ngram_type}_count']]))
        vocab = np.array(list(vocab))
        self.nvocab = len(vocab)
        self.word2idx = {word:idx for idx, word in enumerate(vocab)}
        self.vocab = vocab
    def get_dtm_by_cctype(self, cc_type:str):
        '''
        cc_type: 'present' or 'qa'
        '''

        text_col = f'text_{cc_type}_{self.ngram_type}_count'
        
        n_nonzero = sum(len(d) for d in cc_ngram[text_col])

        # make a list of document names
        # the order will be the same as in the dict
        vocab_sorter = np.argsort(self.vocab)    # indices that sort "vocab"

        data = np.empty(n_nonzero, dtype=np.intc)     # all non-zero term frequencies at data[k]
        rows = np.empty(n_nonzero, dtype=np.intc)     # row index for kth data item (kth term freq.)
        cols = np.empty(n_nonzero, dtype=np.intc)     # column index for kth data item (kth term freq.)

        ind = 0     # current index in the sparse matrix data

        # fill dtm
        for doc_i, row in tqdm(enumerate(cc_ngram.itertuples()), total=len(cc_ngram), desc=f'Building {cc_type.upper()} dtm'):
            # find indices into  such that, if the corresponding elements in  were
            # inserted before the indices, the order of  would be preserved
            # -> array of indices of  in 
            unique_indices = vocab_sorter[np.searchsorted(self.vocab, list(getattr(row, text_col).keys()), sorter=vocab_sorter)]

            # count the unique terms of the document and get their vocabulary indices
            counts = np.array(list(getattr(row, text_col).values()), dtype=np.int32)

            n_vals = len(unique_indices)  # = number of unique terms
            ind_end = ind + n_vals  #  to  is the slice that we will fill with data

            data[ind:ind_end] = counts                  # save the counts (term frequencies)
            cols[ind:ind_end] = unique_indices            # save the column index: index in 
            rows[ind:ind_end] = np.repeat(doc_i, n_vals)  # save it as repeated value

            ind = ind_end  # resume with next document -> add data to the end
        
        # final dtm
        dtm = coo_matrix((data, (rows, cols)), shape=(self.ndoc, self.nvocab), dtype=np.intc)
        if cc_type=='present': self.dtm_present = dtm
        if cc_type=='qa': self.dtm_qa = dtm
        if cc_type=='all': self.dtm_all = dtm
        
        return self
    
    def get_dtm_all(self):
        self.get_dtm_by_cctype('present')
        self.get_dtm_by_cctype('qa')
        self.get_dtm_by_cctype('all')
        return self

corp_allgram = CCCorpus('allgram').get_dtm_all()
sv('corp_allgram')
