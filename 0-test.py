# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.9.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Init

# +
# import tensorflow as tf
import torch
import pytorch_lightning as pl
import torch.nn.functional as F
import torch.optim as optim
import pyarrow.feather as feather

from torch import nn
from torch.utils.data import Dataset, DataLoader

# set random seed
torch.backends.cudnn.deterministic = False;
torch.backends.cudnn.benchmark = True;
torch.backends.cudnn.enabled = True


# -

# # def Data

# Define Dataset
class CCDataset(Dataset):
    
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        target = self.data.iloc[idx,0]
        features = self.data.iloc[idx,1:].to_numpy()

        return torch.tensor(target, dtype=torch.float32), \
               torch.tensor(features, dtype=torch.float32)


# then define DataModule
class CCDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        
    # Dataset
    def setup(self):
        # read the train and test dataset
        targets_train = feather.read_feather('targets_train.feather')
        targets_val = feather.read_feather('targets_test.feather')
        
        self.train_dataset = CCDataset(targets_train)
        self.val_dataset = CCDataset(targets_val)

    # DataLoader
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=64, 
                          shuffle=True, drop_last=False, num_workers=2,
                          pin_memory=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=64,
                          num_workers=2, pin_memory=True,
                          drop_last=False)


# # def Model

class Model(pl.LightningModule):
    '''Mainly define the `*_step_end` methods
    '''
    def __init__(self):
        super().__init__()
        
        # dropout layers
        self.dropout_1 = nn.Dropout(0.5)
        
        # fc layers
        self.fc_1 = nn.Linear(15, 16)
        self.fc_2 = nn.Linear(16, 1)
        
    def shared_step(self, batch):
        t, x = batch
        x = self.dropout_1(F.relu(self.fc_1(x)))
        y = self.fc_2(x) # (N, 1)    
        
        return y.squeeze(), t
        
    # train step
    def training_step(self, batch, idx):
        y, t = self.shared_step(batch)
        return {'y': y, 't': t}
        
    # validation step
    def validation_step(self, batch, idx):
        y, t = self.shared_step(batch)
        return {'y': y, 't': t}
        
    # loss
    def mse_loss(self, y, t):
        return F.mse_loss(y, t)
        
    # def training_step_end
    def training_step_end(self, outputs):
        y = outputs['y']
        t = outputs['t']
        loss = self.mse_loss(y, t)
        
        return {'loss':loss}
    
    # def validation_step_end
    def validation_step_end(self, outputs):
        y = outputs['y']
        t = outputs['t']
        
        return {'y': y, 't': t}
        
    # validation step
    def validation_epoch_end(self, outputs):
        y = torch.cat([x['y'] for x in outputs])
        t = torch.cat([x['t'] for x in outputs])
        
        loss = self.mse_loss(y, t)
        rmse = torch.sqrt(loss)
        self.log('val_rmse', rmse, on_step=False)
        
    # optimizer
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        return optimizer  


# # Run

# +
# ----------------------------
# Initialize model and trainer
# ----------------------------
# checkpoint
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    verbose=True,
    mode='min',
    monitor='val_rmse',
    save_top_k=1)

# trainer
trainer = pl.Trainer(gpus=[0,1], 
                     checkpoint_callback=checkpoint_callback, 
                     accelerator='ddp',
                     min_epochs=10,
                     max_epochs=500)

# loop over windows
torch.manual_seed(42)

# ----------------------------
# fit and test
# ----------------------------
# init model
model = Model()

# create datamodule
datamodule = CCDataModule()
datamodule.setup()

# train the model
trainer.fit(model, datamodule)
