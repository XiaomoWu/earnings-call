import gc
import pyarrow.feather as feather
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tqdm import tqdm
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

class finBertDataset(Dataset):
    def __init__(self, sentence_id, texts):
        self.sentence_id = sentence_id
        self.texts = texts

    def __len__(self):
        return len(self.sentence_id)

    def __getitem__(self, idx):
        return (self.sentence_id[idx], self.texts[idx])

def train(rank, ciq_components_sentencized, world_size, batch_size):

    # ----------------------------
    # Initialization
    # ----------------------------
    
    ROOT_DIR = 'C:/Users/rossz/Onedrive/CC'
    DATA_DIR = f'{ROOT_DIR}/data'

    # Initiazlie random seed
    torch.manual_seed(42)

    # Initialize process
    print(f'Initializing rank {rank}...')
    dist.init_process_group(                                   
    	backend='gloo',                                         
   		init_method=f'file://{DATA_DIR}/Embeddings/ddp.log',                                   
    	world_size=world_size,                              
    	rank=rank                                               
    ) 

    # ----------------------------
    # Create Dataset/DataLoader
    # ----------------------------

    sentence_ids = ciq_components_sentencized['sentence_id']
    texts = ciq_components_sentencized['text']

    dataset = finBertDataset(sentence_ids, texts)
    sampler = DistributedSampler(dataset,
                                 shuffle=False,
                                 num_replicas=world_size,
                                 rank=rank)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                            sampler=sampler, num_workers=0, pin_memory=True)

    # ----------------------------
    # Create Model
    # ----------------------------
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('./data/finBERT', return_dict=True)
    model.to(rank)
    model = DDP(model, device_ids=[rank])
    
    # ----------------------------
    # Start Predicting!
    # ----------------------------

    output = {}
    n_batches = len(dataloader)
    for batch_idx, (sids, texts) in enumerate(dataloader):
        if batch_idx%1000==0:
            print(f'{batch_idx}/{n_batches}')

        # tokenize
        tokens = tokenizer(texts,return_tensors='pt', padding=True, truncation=True, max_length=384)
        tokens = tokens.to(rank)
        
        # get mask
        # - the 1st and last token are special token, set to zero
        # - devide mask by (seq_len-2)
        mask = tokens.attention_mask.float() # (B, S)
        valid_seq_len = torch.sum(mask==1, dim=1)

        for i, l in enumerate(valid_seq_len):
            mask[i, [0, l-1]] = 0
            mask[i] /= (l-2)

        mask = mask.unsqueeze(-1) # (B, S, 1)

        # compute embedding
        embedding = model(**tokens).last_hidden_state.transpose(-1, 1) # (B, E, S)
        embedding_pool = torch.bmm(embedding, mask).squeeze().detach().to('cpu') # (B, E)   
        if len(embedding_pool.shape)==1:
            embedding_pool = embedding_pool.unsqueeze(0)

        del embedding, mask, tokens

        for _ in range(len(sids)):
            output[sids[_]] = {'seq_len': valid_seq_len[_].item(), 
                               'embedding': embedding_pool[_,...]}



        # Save checkpoint
        if batch_idx>0 and ((batch_idx%(n_batches//25)==0) or (batch_idx>=(n_batches-1))):

            refresh_cuda_memory()

            # Create Model
            tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
            model = BertModel.from_pretrained('./data/finBERT', return_dict=True)
            model.to(rank)
            model = DDP(model, device_ids=[rank])

            sv_name = f'preembeddings_finbert_nocls_rank{rank}_{batch_idx}.pt'
            print(f'Saving to {sv_name}')
            torch.save(output, f'./data/Embeddings/{sv_name}')
            output = {}
    
    # torch.save(output, f'./data/Embeddings/preembeddings_finbert_withoutcls.pt')

def main():
    # Hyper parameters
    batch_size = 16
    world_size = 2

    # load sentences
    ciq_components_sentencized = feather.read_feather(f'data/ciq_components_sentencized.feather')
    print(f'N sentences: {len(ciq_components_sentencized)}')

    mp.spawn(train, 
             nprocs=world_size, 
             args=(ciq_components_sentencized, world_size, batch_size))


if __name__ == '__main__':
    main()