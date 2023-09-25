#!/usr/bin/env python
# coding: utf-8

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


llogger = CSVLogger("emotions", 'classification-lm')


data_path = 'IEMOCAP_full_release/Session*/sentences/wav/Ses*/*.wav'
files = glob.glob(data_path)
file_name = [x.split(os.sep)[-1].split('.')[0] for x in files]


csv_file_paths = 'IEMOCAP_full_release/Session*/dialog/EmoEvaluation/Categorical/Ses*.txt'


dialogues_path_reg = 'IEMOCAP_full_release/Session*/dialog/transcriptions/Ses*.txt'
dialogues_paths = glob.glob(dialogues_path_reg)


csv_files = glob.glob(csv_file_paths)





def get_name_category_from_path(path):
    file_name_list = []
    emotion_list = []
    with open(path, 'r') as f:
        for line in f.readlines():
            file_name_list.append(re.findall('Ses.*[A-Z][0-9]{3}', line)[0])
            emotion_list.append(re.findall(':.*;', line)[0][1:-1].lower())
    return file_name_list, emotion_list
        


def get_name_dial_from_path(path):
    file_name_list = []
    dial_list = []
    with open(path, 'r') as f:
        for line in f.readlines():
            file_name_list.append(re.findall('Ses.*[A-Z][0-9]{3}', line)[0])
            dial_list.append(re.findall(':.*', line)[0][1:])
    return file_name_list, dial_list


file_list = []
category_list = []
for file in csv_files:
    f, e = get_name_category_from_path(file)
    file_list.extend(f)
    category_list.extend(e)
cat_df = pd.DataFrame(
    {
        'file':file_list,
        'category':category_list
    }
)


file_list = []
dial_list = []
for file in dialogues_paths:
    try:
        f, d = get_name_dial_from_path(file)
        file_list.extend(f)
        dial_list.extend(d)
    except Exception as e:
        continue
dial_df = pd.DataFrame(
    {
        'file':file_list,
        'dial':dial_list
    }
)


cat_df['category'] = cat_df['category'].str.replace(';.*','')


cat_df['category'].shape


cat_df[cat_df['file']=='Ses04F_script02_2_F000']


meta_df = pd.merge(cat_df, dial_df, on = 'file').groupby(['file','category','dial']).count().reset_index()


meta_df['category'].unique()


from torch.utils.data import Dataset
import glob
from transformers import BertTokenizer, BertModel

class AudioDataset(Dataset, ):
    """
        Custom dataset class for loading audio dataset.
        meta_df: dataframe containing file_name (without 
                the .wav extension) and the category (labels) 
        directory: regular expression for the directory to look 
                for wav files. (e.g. /dataset/speech/*.wav)
    """
    
    def __init__(self, meta_df, directory,modality = 'all', **kwargs):
        self.meta_df = meta_df
        self.directory = directory
        self.audio_path_list = glob.glob(directory)
        category = self.meta_df['category'].unique()
        self.t_dict = dict(zip(category,range(len(category))))
        self.kwargs = kwargs
        self.modality = modality
    
    def __len__(self):
        return len(self.meta_df)
    
    def __getitem__(self, idx):
        audio_path = self.audio_path_list[idx]
        audio_name = os.path.basename(audio_path).split('.')[0]
        dialogue = self.meta_df.loc[idx,['dial']].values[0]
        targets = self.meta_df.loc[idx,[ 'category']].values[0]
        audio_feature = None
        encoded_text = None
        if self.modality == 'audio' :
            signal, sr = torchaudio.load(audio_path, )
            signal =  signal[0]
            feature_extractor = AutoFeatureExtractor.from_pretrained("ast-finetuned-audioset-10-10-0.4593")
            audio_feature = feature_extractor(signal, sampling_rate=sr, return_tensors="pt")
            
            return audio_feature, self.t_dict[targets]
            
        elif self.modality == 'text':
            
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            encoded_text = tokenizer(dialogue,return_tensors="pt",  padding="max_length", max_length=50, add_special_tokens=True, truncation=True,)
            return  encoded_text, self.t_dict[targets]
        
        else:
            signal, sr = torchaudio.load(audio_path, )
            signal =  signal[0]
            feature_extractor = AutoFeatureExtractor.from_pretrained("ast-finetuned-audioset-10-10-0.4593")

            audio_feature = feature_extractor(signal, sampling_rate=sr, return_tensors="pt")
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            encoded_text = tokenizer(dialogue,return_tensors="pt",  padding="max_length", max_length=50, add_special_tokens=True, truncation=True,)
            return audio_feature, encoded_text, self.t_dict[targets]
    


asset = AudioDataset(meta_df,data_path)


# next(iter(dataloader))[1]['input_ids'].shape


class LighteningModel(pl.LightningModule):
    def __init__(self, input_size=48000, num_classes=10, hidden_size = 200, num_heads = 5, num_layers_tx  = 2):
        super(LighteningModel, self).__init__()
        self.train_r2 = torchmetrics.R2Score()
        self.val_r2 = torchmetrics.R2Score()
        self.train_auc = torchmetrics.classification.AUROC(task = 'multiclass', num_classes=num_classes)
        self.val_auc = torchmetrics.classification.AUROC(task = 'multiclass', num_classes=num_classes)
        
        self.val_f1 = torchmetrics.classification.F1Score(task = 'multiclass', num_classes=num_classes)
        self.train_f1 = torchmetrics.classification.F1Score(task = 'multiclass', num_classes=num_classes)
        
        
        feature_extractor = AutoFeatureExtractor.from_pretrained("ast-finetuned-audioset-10-10-0.4593")
        self.sp_model = ASTForAudioClassification.from_pretrained("ast-finetuned-audioset-10-10-0.4593", return_dict=False)
        
        self.bert_model = BertModel.from_pretrained("bert-base-uncased")
        
        for param in self.sp_model.parameters():
            param.requires_grad = False
        self.sp_model.classifier.dense = nn.Linear(768,256)
        
        for param in self.bert_model.parameters():
            param.requires_grad = False
            
        for param in self.bert_model.encoder.layer[-1:].parameters():
            param.requires_grad = True
                                                                                                                        
        self.fc1 = nn.Linear(1024, num_classes)
        self.dropout = nn.Dropout(.7) 
        self.relu = nn.ReLU()
    def forward(self,x,y):
        x = x['input_values'].view(x['input_values'].size(0), 1024,128)
        input_bert = y['input_ids'].view(y['input_ids'].size(0), 50)
        atten_bert =  y['attention_mask'].view(y['attention_mask'].size(0), 50)
#         y = y['input_ids']
#         print(y.shape)
        sp = self.sp_model(x, return_dict=False)[0] #256
        bert,pool = self.bert_model(input_ids = input_bert, attention_mask = atten_bert, return_dict = False)
        bert = self.dropout(pool)
        sp = self.dropout(sp)

        out = torch.concat([bert, sp], axis = 1)
        out = self.dropout(out)
        out = self.fc1(out)
        
        return out
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.0001, weight_decay=1e-3)
        return optimizer
    
    
    def training_step(self, train_batch, batch_idx):
        signal, dial, labels = train_batch
#         signal = signal['input_values'].view(signal['input_values'].size(0), 1024,128).shape
        labels =labels.float().to('cuda:0')
        outputs = self(signal, dial).to('cuda:0')#.argmax(1).float()
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(outputs, labels.long())
        self.train_auc(outputs, labels.int())
        self.train_f1(outputs, labels.int())
        
        self.log('loss', loss, on_step=False, on_epoch=True)
        self.log('train_auc', self.train_auc, on_step=False, on_epoch=True)
        self.log('train_f1', self.train_f1, on_step=False, on_epoch=True)
        
        
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        signal ,dial, labels = val_batch
        labels = labels.float().to('cuda:0')
#         signal = signal['input_values'].view(signal['input_values'].size(0), 1024,128).shape
        
        outputs = self(signal, dial).to('cuda:0')#.argmax(1).float()
        criterion = torch.nn.CrossEntropyLoss()
        
        loss = criterion(outputs, labels.long())
        self.val_auc(outputs, labels.int())
        self.val_f1(outputs, labels.int())
        
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.log('val_auc', self.val_auc, on_step=False, on_epoch=True)
        self.log('val_f1', self.val_f1, on_step=False, on_epoch=True)



model = LighteningModel()


next(model.parameters()).is_cuda


BATCH_SIZE = 16


dataset = AudioDataset(meta_df,data_path, modality = 'all')


len(dataset)


meta_df.shape


train_set, val_set = torch.utils.data.random_split(dataset, [0.8, 0.2],)


# sampler = WeightedRandomSampler(sample_weights, int(len(train_set)*1.5), replacement=True)
train_dataloader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle = True, num_workers=8, drop_last=True, )
test_dataloader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, drop_last=True)


trainer = pl.Trainer( max_epochs=500, gradient_clip_val=0, logger = llogger, )


trainer.fit(model, train_dataloader, test_dataloader, )









