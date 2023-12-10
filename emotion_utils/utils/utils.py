"""
    author:siddhi47
    desc: Utilities for multimodal(audio and text).
"""

import re
import os
import glob
import argparse
import torchaudio
import pandas as pd
from torch.utils.data import Dataset
from transformers import BertTokenizer
from transformers import AutoFeatureExtractor, ASTFeatureExtractor
from speechbrain.pretrained import SepformerSeparation as separator

def get_name_category_from_path(path):
    file_name_list = []
    emotion_list = []
    with open(path, "r") as f:
        for line in f.readlines():
            try:
                file_name_list.append(re.findall("Ses.*[A-Z][0-9]{3}", line)[0])
                emotion_list.append(re.findall(":.*;", line)[0][1:-1].lower())
            except:
                continue
    return file_name_list, emotion_list


def get_name_dial_from_path(path):
    file_name_list = []
    dial_list = []
    with open(path, "r") as f:
        for line in f.readlines():
            try:
                file_name_list.append(re.findall("Ses.*[A-Z][0-9]{3}", line)[0])
                dial_list.append(re.findall(":.*", line)[0][1:])
            except:
                continue
    return file_name_list, dial_list

def get_name_mocap_path_from_path(path):
    file_name_list = []
    mocap_list = []
    for file in glob.glob(path):
        file_name_list.append(os.path.basename(file).split(".")[0])
        mocap_list.append(file)

    return file_name_list, mocap_list

def get_meta(path_regex, label="category"):
    dial_category = {
        "category": get_name_category_from_path,
        "dial": get_name_dial_from_path,
    }
    file_list = []
    label_list = []
    for file in glob.glob(path_regex):
        f, e = dial_category[label](file)
        file_list.extend(f)
        label_list.extend(e)
    return pd.DataFrame({"file": file_list, label: label_list})

def add_mocap(df, path):
    file_name_list, mocap_list = get_name_mocap_path_from_path(path)
    df["mocap"] = df["file"].apply(lambda x: mocap_list[file_name_list.index(x)])
    return df

class AudioDataset(
    Dataset,
):
    """
    Custom dataset class for loading audio dataset.
    meta_df: dataframe containing file_name (without
            the .wav extension) and the category (labels)
    directory: regular expression for the directory to look
            for wav files. (e.g. /dataset/speech/*.wav)
    """

    def __init__(self, meta_df, directory, modality="multimodal", **kwargs):
        self.meta_df = meta_df
        self.directory = directory
        self.audio_path_list = glob.glob(directory)
        category = self.meta_df["category"].unique()
        self.t_dict = dict(zip(category, range(len(category))))
        self.kwargs = kwargs
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.modality = modality
        #self.audio_feature_extractor = AutoFeatureExtractor.from_pretrained(
        #        "ast-finetuned-audioset-10-10-0.4593", #return_tensors = 'pt',
        #        max_length = 1024
        #    )
        self.audio_feature_extractor = ASTFeatureExtractor.from_pretrained(
                "ast-finetuned-audioset-10-10-0.4593", #return_tensors = 'pt',
                max_length = 1024,
                return_attention_mask = True

            )

    def __len__(self):
        return len(self.meta_df)

    def resample_audio(self,audio,sr, target_length = 5):
        audio_length_seconds = audio.shape[1]/sr
        if audio_length_seconds<target_length:
            multiply_num = round(target_length/audio_length_seconds)

            audio = torch.concat(multiply_num*[audio],dim = 1)
        return audio

    def save_clean_audio(self, audio_path, clean_audio_path):
        separator.separate_file(
            audio_path,
            savedir=os.path.dirname(audio_path),
            saved_filename=os.path.basename(audio_path).split(".")[0],
        )
        os.rename(audio_path, clean_audio_path)

    def __getitem__(self, idx):
        audio_path = self.audio_path_list[idx]
        clean_audio_path = os.path.dirname(audio_path) + "/clean/" + os.path.basename(audio_path)
        audio_name = os.path.basename(audio_path).split(".")[0]
        targets = self.meta_df.loc[idx, ["category"]].values[0]
        audio_feature = None
        encoded_text = None
        if self.modality in [ "audio", "audio1"]:
            
            #if os.path.exists(clean_audio_path):
                #audio_path = clean_audio_path
            #else:
                #self.save_clean_audio(audio_path, clean_audio_path)

            signal, sr = torchaudio.load(
                audio_path,
            )
            #signal = self.resample_audio(signal[0], sr)
            signal = signal[0]


            audio_feature = self.audio_feature_extractor(
                signal, sampling_rate=sr, return_tensors="pt"
            )
        

            return audio_feature, self.t_dict[targets]

        elif self.modality == "text":
            dialogue = self.meta_df.loc[idx, ["dial"]].values[0]

            encoded_text = self.tokenizer(
                dialogue,
                return_tensors="pt",
                padding="max_length",
                max_length=50,
                add_special_tokens=True,
                truncation=True,
            )
            return encoded_text, self.t_dict[targets]
            
        else:
            dialogue = self.meta_df.loc[idx, ["dial"]].values[0]

            signal, sr = torchaudio.load(
                audio_path,
            )
            signal = signal[0]
            

            audio_feature = self.audio_feature_extractor(
                signal, sampling_rate=sr, return_tensors="pt"
            )
            encoded_text = self.tokenizer(
                dialogue,
                return_tensors="pt",
                padding="max_length",
                max_length=50,
                add_special_tokens=True,
                truncation=True,
            )
            return audio_feature, encoded_text, self.t_dict[targets]


def ArgParser():
    parser = argparse.ArgumentParser(description="Emotion Recognition")
    parser.add_argument(
        "--config",
        type=str,
        default="config.ini",
        help="config file path (default: config.ini)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        help="train or test (default: train)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="audio",
        help="audio, text or multimodal (default: audio)",
        choices=["audio", "text", "multimodal",'face', 'audio1', 'multimodalPre'],
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="batch size (default: 32)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="number of workers (default: 4)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=100,
        help="max epochs (default: 100)",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="emotion_utils/logs",
        help="log directory (default: emotion_utils/logs)",
    )

    parser.add_argument(
        "--sample_frac",
        type=float,
        default=1.0,
        help="sample fraction for weighted sampler. (default: 1.0)",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=int,
        default=None,
        help="From which version to resume training.",
    )
    return parser.parse_args()
