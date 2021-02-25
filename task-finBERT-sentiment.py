with open("/home/yu/OneDrive/App/Settings/jupyter + R + Python/python_startup.py", 'r') as f:
    exec(f.read())


import os
os.chdir('/home/yu/OneDrive/CC')

import gc
import pyarrow.feather as feather
import torch
import torch.distributed as dist
import torch.nn.functional as F


from functools import partial
from tqdm import tqdm
from torch.multiprocessing import Process
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, AutoModelForSequenceClassification

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
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

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

            # tokenize
            tokens = tokenizer(texts,return_tensors='pt', padding=True, truncation=True, max_length=384)
            tokens = tokens.to(rank)
            
            # compute embedding
            logits = model(**tokens).logits
            y = F.softmax(logits,dim=1).to('cpu')

            for _, (tid, sid) in enumerate(zip(tids, sids)):
                output[f'{tid}_{sid}'] = y[_]

    torch.save(output, f'data/Embeddings/sentiment_finbert_rank{rank}.pt')

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