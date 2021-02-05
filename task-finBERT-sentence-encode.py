with open("/home/yu/OneDrive/App/Settings/jupyter + R + Python/python_startup.py", 'r') as f:
    exec(f.read())


import os
os.chdir('/home/yu/OneDrive/CC')

import gc
import pyarrow.feather as feather
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from functools import partial
from tqdm import tqdm
from torch.multiprocessing import Process, Queue
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import BertModel, BertTokenizer, BertTokenizerFast

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

# helper: print elapsed time (given start and end)
# def elapsed_time(start, end):
#     hours, rem = divmod(end-start, 3600)
#     minutes, seconds = divmod(rem, 60)
#     print(f'{int(hours)}h {int(minutes)}min {seconds:.1f}s')

class finBertDataset(Dataset):
    def __init__(self, sents):
        self.sents = sents
        '''
        assert len(transcriptids) == len(sentenceids) == len(texts)
        self.transcriptids = transcriptids
        self.sentenceid = sentenceids
        self.texts = texts
        self.n_samples = len(transcriptids)
        '''

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, idx):
        transcriptid, _, sentenceid, text = self.sents[idx]
        return (transcriptid, sentenceid, text)

        
def train(rank, world_size, sents, batch_size):

    # ----------------------------
    # Initialization
    # ----------------------------
    
    ROOT_DIR = '/home/yu/OneDrive/CC'
    DATA_DIR = f'{ROOT_DIR}/data'

    # Initiazlie random seed
    torch.manual_seed(42)

    # Initialize process
    print(f'Finished initializing rank {rank}')
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group(
    	backend='nccl',
    	world_size=world_size,
    	rank=rank) 

    # ----------------------------
    # Create Dataset/DataLoader
    # ----------------------------

    dataset = finBertDataset(sents)
    sampler = DistributedSampler(dataset,
                                 shuffle=False,
                                 num_replicas=world_size,
                                 rank=rank)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                            sampler=sampler, num_workers=2, pin_memory=True)

    # ----------------------------
    # Create Model
    # ----------------------------
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('./data/finBERT', return_dict=True)
    model.to(rank)
    model.eval()
    model = DDP(model, device_ids=[rank])
    
    # ----------------------------
    # Start Predicting!
    # ----------------------------

    output = {}
    if rank==0:
        print(f'batch_size={batch_size}')

    with torch.no_grad():
        for batch_idx, (tids, sids, texts) in enumerate(tqdm(dataloader, f'rank{rank}', smoothing=0, mininterval=0.5, position=rank)):

            # if batch_idx%1000==0 and dist.get_rank()==0:
            #     print(f'{batch_idx}/{n_batches} batches, {pretty_time_delta(time.perf_counter()-t0)} elapsed')

            # tokenize
            tokens = tokenizer(texts,return_tensors='pt', padding=True, truncation=True, max_length=384)
            tokens = tokens.to(rank)
            
            # get mask
            mask = tokens.attention_mask.float() # (B, S)
            valid_seq_len = torch.sum(mask==1, dim=1) # (B,)

            '''
            # Option 1:
            # - the 1st and last token are special token, set to zero
            # - devide mask by (seq_len-2)
            for i, l in enumerate(valid_seq_len):
                mask[i, [0, l-1]] = 0
                mask[i] /= (l-2)
                
            # Option 2:
            # - all tokens (including [CLS] and [EOS]) are preserved
            # - devide mask by (seq_len)
            for i, l in enumerate(valid_seq_len):
                mask[i] /= l
            mask = mask.unsqueeze(-1) # (B, S, 1)
            '''

            # Option 3:
            # - Only [CLS] is preserved
            # - devide mask by (seq_len-1)
            for i, l in enumerate(valid_seq_len):
                mask[i, [l-1]] = 0
                mask[i] /= (l-1)
            mask = mask.unsqueeze(-1) # (B, S, 1)

            # compute embedding
            embedding = model(**tokens).last_hidden_state.transpose(-1, 1) # (B, E, S)
            embedding_pool = torch.bmm(embedding, mask).squeeze().detach().to('cpu') # (B, E)   
            if len(embedding_pool.shape)==1:
                embedding_pool = embedding_pool.unsqueeze(0)

            del embedding, mask, tokens

            for _, (tid, sid) in enumerate(zip(tids, sids)):
                output[f'{tid}_{sid}'] = {'seq_len': valid_seq_len[_].item(), 'embedding': embedding_pool[_,...]}



            # Save checkpoint
            # if batch_idx>0 and ((batch_idx%(n_batches//25)==0) or (batch_idx>=(n_batches-1))):

            #     refresh_cuda_memory()

            #     # Create Model
            #     tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
            #     model = BertModel.from_pretrained('./data/finBERT', return_dict=True)
            #     model.to(rank)
            #     model = DDP(model, device_ids=[rank])

            #     sv_name = f'preembeddings_finbert_with_special_tokens_rank{rank}_{batch_idx}.pt'
            #     print(f'Saving to {sv_name}')
            #     torch.save(output, f'./data/Embeddings/{sv_name}')
            #     output = {}
        
    torch.save(output, f'data/Embeddings/preembeddings_finbert_onlyCLS_rank{rank}.pt')

    # print(f'rank {rank}: {list(output.keys())}')

    # clean up
    dist.destroy_process_group()


# ---------------------------- 
# Run Model 
# ----------------------------

if __name__ == '__main__':
    # Hyper parameters
    batch_size = 64
    world_size = 2

    # load data
    print('Loading sentence data...')
    ld('sents_sp500', ldname='sents')

    sents = sents
    print(f'N sentences: {len(sents)}')

    # Create process!
    processes = []

    for rank in range(world_size):
        p = Process(target=train, args=(rank, world_size, sents, batch_size))
        p.start()
        processes.append(p)

    for process in processes:
        p.join()

    # print(list(outputs.keys()))
    