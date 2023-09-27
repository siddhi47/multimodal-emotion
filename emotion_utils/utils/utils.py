"""
    author:siddhi47
    desc: Utilities for multimodal(audio and text).
"""

import re
import os
import glob
import torchaudio
import pandas as pd
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertModel
from transformers import AutoFeatureExtractor, ASTForAudioClassification


def get_name_category_from_path(path):
    file_name_list = []
    emotion_list = []
    with open(path, 'r') as f:
        for line in f.readlines():
            try:
                file_name_list.append(re.findall('Ses.*[A-Z][0-9]{3}', line)[0])
                emotion_list.append(re.findall(':.*;', line)[0][1:-1].lower())
            except:
                continue
    return file_name_list, emotion_list
        
    
def get_name_dial_from_path(path):
    file_name_list = []
    dial_list = []
    with open(path, 'r') as f:
        for line in f.readlines():
            try:
                file_name_list.append(re.findall('Ses.*[A-Z][0-9]{3}', line)[0])
                dial_list.append(re.findall(':.*', line)[0][1:])
            except: 
                continue
    return file_name_list, dial_list

def get_meta(path_regex, label = 'category'):
    dial_category  = {
        'category':get_name_category_from_path,
        'dial': get_name_dial_from_path
    }
    file_list = []
    label_list = []
    for file in glob.glob(path_regex):
        f, e = dial_category[label](file)
        file_list.extend(f)
        label_list.extend(e)
    return pd.DataFrame(
        {
            'file':file_list,
            label:label_list
        }
    )
    
    

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
            dialogue = self.meta_df.loc[idx,['dial']].values[0]
            
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            encoded_text = tokenizer(
                dialogue,return_tensors="pt",
                padding="max_length",
                max_length=50,
                add_special_tokens=True,
                truncation=True,
            )
            return  encoded_text, self.t_dict[targets]
        
        else:
            dialogue = self.meta_df.loc[idx,['dial']].values[0]
            
            signal, sr = torchaudio.load(audio_path, )
            signal =  signal[0]
            feature_extractor = AutoFeatureExtractor.from_pretrained("ast-finetuned-audioset-10-10-0.4593")

            audio_feature = feature_extractor(signal, sampling_rate=sr, return_tensors="pt")
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            encoded_text = tokenizer(
                dialogue,return_tensors="pt",
                padding="max_length",
                max_length=50,
                add_special_tokens=True,
                truncation=True,
            )
            return audio_feature, encoded_text, self.t_dict[targets]
            


