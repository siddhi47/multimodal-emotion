import warnings
warnings.filterwarnings('ignore')

import torch
import torchaudio
import os
import librosa
from IPython.display import Audio
import librosa
from torchaudio.utils import download_asset
import cv2
import glob
import pandas as pd


from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
from torch import nn
import torch
import torch.optim as optim
import lightning.pytorch as pl
from torchaudio.transforms import *
import numpy as np
import torchmetrics
from lightning.pytorch.loggers import CSVLogger
from torch.nn.utils.rnn import pad_sequence
import re
from transformers import AutoFeatureExtractor, ASTForAudioClassification
from torch.utils.data import DataLoader
from emotion_utils.utils.utils import *
from torch.utils.data import Dataset
import glob


import configparser
config = configparser.ConfigParser()
config.read("config.ini")
BATCH_SIZE = int(config['TRAINING']['BATCH_SIZE'])
MAX_EPOCH = int(config['TRAINING']['MAX_EPOCH'])

llogger = CSVLogger("emotions", 'classification')


data_path = 'IEMOCAP_full_release/Session*/sentences/wav/Ses*/*.wav'
csv_file_path_reg = 'IEMOCAP_full_release/Session*/dialog/EmoEvaluation/Categorical/Ses*.txt'
dialogues_path_reg = 'IEMOCAP_full_release/Session*/dialog/transcriptions/Ses*.txt'

cat_df = get_meta(csv_file_path_reg, 'category')
dial_df  =get_meta(dialogues_path_reg, 'dial')

meta_df = pd.merge(cat_df, dial_df, on = 'file').groupby(['file','dial']).head(1).reset_index().drop(columns = ['index'])
meta_df['category'] = meta_df['category'].str.replace(';.*','', regex=True)



class LighteningModel(pl.LightningModule):
    def __init__(self, input_size=48000, num_classes=10, hidden_size = 200, num_heads = 5, num_layers_tx  = 2):
        super(LighteningModel, self).__init__()
        self.train_r2 = torchmetrics.R2Score()
        self.val_r2 = torchmetrics.R2Score()
        self.train_auc = torchmetrics.classification.AUROC(task = 'multiclass', num_classes=num_classes)
        self.val_auc = torchmetrics.classification.AUROC(task = 'multiclass', num_classes=num_classes)
        
        feature_extractor = AutoFeatureExtractor.from_pretrained("ast-finetuned-audioset-10-10-0.4593")
        self.sp_model = ASTForAudioClassification.from_pretrained("ast-finetuned-audioset-10-10-0.4593", return_dict=False)
        for param in self.sp_model.parameters():
            param.requires_grad = False
        self.sp_model.classifier.dense = nn.Linear(768,527)
        
        self.fc1 = nn.Linear(527, num_classes)
        self.dropout = nn.Dropout(.5) 
        self.relu = nn.ReLU()
    def forward(self,x,):
        x = x['input_values'].view(x['input_values'].size(0), 1024,128)
        
        out = self.sp_model(x, return_dict=False)[0] #256
        out = self.dropout(out)
        out = self.fc1(out)
#         out = self.sigmoid(out)
        return out
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.0001, weight_decay=1e-3)
        return optimizer
    
    
    def training_step(self, train_batch, batch_idx):
        signal, labels = train_batch
        labels =labels.float().to('cuda:0')
        outputs = self(signal).to('cuda:0')#.argmax(1).float()
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(outputs, labels.long())
        self.train_auc(outputs, labels.int())
        self.log('loss', loss, on_step=False, on_epoch=True)
        self.log('train_auc', self.train_auc, on_step=False, on_epoch=True)
        
        
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        signal ,labels = val_batch
        labels = labels.float().to('cuda:0')
#         signal = signal['input_values'].view(signal['input_values'].size(0), 1024,128).shape
        
        outputs = self(signal).to('cuda:0')#.argmax(1).float()
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(outputs, labels.long())
        self.val_auc(outputs, labels.int())
        
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.log('val_auc', self.val_auc, on_step=False, on_epoch=True)




model = LighteningModel()

dataset = AudioDataset(meta_df,data_path, modality='audio')
train_set, val_set = torch.utils.data.random_split(dataset, [0.8, 0.2],)


train_dataloader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle = True, num_workers=8, drop_last=True, )
test_dataloader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, drop_last=True)

trainer = pl.Trainer( max_epochs=MAX_EPOCH, gradient_clip_val=0, logger = llogger)

trainer.fit(model, train_dataloader, test_dataloader)
