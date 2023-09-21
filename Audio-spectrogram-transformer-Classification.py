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


llogger = CSVLogger("emotions", 'classification')



from torch.utils.data import Dataset
import glob

class AudioDataset(Dataset):
    """
        Custom dataset class for loading audio dataset.
        meta_df: dataframe containing file_name (without 
                the .wav extension) and the category (labels) 
        directory: regular expression for the directory to look 
                for wav files. (e.g. /dataset/speech/*.wav)
    """
    
    def __init__(self, meta_df, directory, **kwargs):
        self.meta_df = meta_df
        self.directory = directory
        self.audio_path_list = glob.glob(directory)
        category = self.meta_df['category'].unique()
        self.t_dict = dict(zip(category,range(len(category))))
        print(category)
        self.kwargs = kwargs
    
    def __len__(self):
        return len(self.audio_path_list)
    
    def __getitem__(self, idx):
        audio_path = self.audio_path_list[idx]
        audio_name = os.path.basename(audio_path).split('.')[0]
        targets = self.meta_df.loc[idx,[ 'category']].values[0]
        signal, sr = torchaudio.load(audio_path,  )
        
        feature_extractor = AutoFeatureExtractor.from_pretrained("ast-finetuned-audioset-10-10-0.4593")
        signal =  signal[0]
        return feature_extractor(signal, sampling_rate=16000, return_tensors="pt"), self.t_dict[targets]
    
    
    


data_path = 'IEMOCAP_full_release/Session*/sentences/wav/Ses*/*.wav'
csv_file_paths = 'IEMOCAP_full_release/Session*/dialog/EmoEvaluation/Categorical/Ses*.txt'

csv_files = glob.glob(csv_file_paths)


def get_name_category_from_path(path):
    file_name_list = []
    emotion_list = []
    with open(path, 'r') as f:
        for line in f.readlines():
            file_name_list.append(re.findall('Ses.*[A-Z][0-9]{3}', line)[0])
            emotion_list.append(re.findall(':.*;', line)[0][1:-1].lower())
    return file_name_list, emotion_list
        

file_list = []
category_list = []
for file in csv_files:
    f, e = get_name_category_from_path(file)
    file_list.extend(f)
    category_list.extend(e)
meta_df = pd.DataFrame(
    {
        'file':file_list,
        'category':category_list
    }
)


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
BATCH_SIZE = 32

dataset = AudioDataset(meta_df,data_path)
train_set, val_set = torch.utils.data.random_split(dataset, [0.8, 0.2],)


train_dataloader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle = True, num_workers=8, drop_last=True, )
test_dataloader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, drop_last=True)

trainer = pl.Trainer( max_epochs=500, gradient_clip_val=0, logger = llogger)

trainer.fit(model, train_dataloader, test_dataloader)
