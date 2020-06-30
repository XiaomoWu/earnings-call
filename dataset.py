from torch.utils.data import Dataset
from datetime import datetime
from operator import itemgetter
import torch

# Dataset: Txt + Fin-ratio
class CCDataset(Dataset):
    
    def __init__(self, split_window, split_type, text_in_dataset, roll_type, print_window, preembeddings, targets_df, split_df, valid_transcriptids=None):
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
            transcriptids = targets_df[targets_df.ciq_call_date.between(train_start, train_end)].transcriptid.sample(frac=1, random_state=42).tolist()
            transcriptids = transcriptids[:int(len(transcriptids)*0.9)]
            
        if split_type=='val':
            transcriptids = targets_df[targets_df.ciq_call_date.between(train_start, train_end)].transcriptid.sample(frac=1, random_state=42).tolist()
            transcriptids = transcriptids[int(len(transcriptids)*0.9):]

        elif split_type=='test':
            transcriptids = targets_df[targets_df.ciq_call_date.between(test_start, test_end)].transcriptid.tolist()

        self.valid_preemb_keys = set(transcriptids).intersection(set(preembeddings.keys()))
        
        if valid_transcriptids is not None:
            self.valid_preemb_keys = self.valid_preemb_keys.intersection(set(valid_transcriptids))
        
        # self attributes
        self.text_in_dataset = text_in_dataset
        if text_in_dataset:
            self.preembeddings = preembeddings
        self.targets_df = targets_df
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
        
        # all of the following targests are
        # of type `numpy.float64`
        docid = targets.docid
        
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
        revision = targets.revision
        inflow = targets.inflow
        
        if self.text_in_dataset:
            # inputs: preembeddings
            embeddings = self.preembeddings[transcriptid]
            
            return car_0_30, transcriptid, embeddings, alpha, car_m1_m1, car_m2_m2, car_m30_m3, \
                   sest, sue, numest, sstdest, smedest, \
                   mcap, roa, bm, debt_asset, volatility, volume, inflow, revision
        else:
            return docid, \
                   torch.tensor(car_0_30,dtype=torch.float32), \
                   torch.tensor([alpha, car_m1_m1, car_m2_m2, car_m30_m3, sest, sue, numest, sstdest, smedest, mcap, roa, bm, debt_asset, volatility, volume], dtype=torch.float32)