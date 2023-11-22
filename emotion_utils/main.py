"""
    author:siddhi47
    desc: Contains the main function for the emotion recognition (using text, audio and both)
"""

import warnings

warnings.filterwarnings("ignore")
import torch
import pandas as pd
import torch
import lightning.pytorch as pl
from torchaudio.transforms import *
import numpy as np
from lightning.pytorch.loggers import CSVLogger
import configparser
from torch.utils.data import DataLoader
from emotion_utils.utils.utils import *
from emotion_utils.utils.models import *
from torch.utils.data import WeightedRandomSampler
from lightning.pytorch.callbacks import ModelCheckpoint
torch.set_float32_matmul_precision('medium' )

def get_model(model_name,from_checkpoint = None,  **kwargs):
    """
    Returns the model from model name

    Args:
        model_name (str): Name of the model to be used

    returns:
        model (torch.nn.Module): Model to be used
    """
    model_dict = {
        "audio": AudioSpectrogramModel,
        "audio1": AudioSpectrogramModel1,
        "text": LangModel,
        "multimodal": AudioLangModel,
        "face": FaceModel,
    }
    if model_name not in model_dict:
        raise ValueError(
            f"Model {model_name} not present in the model dictionary. Choose from {model_dict.keys()}"
        )
    if from_checkpoint:
        return model_dict[model_name].load_from_checkpoint(from_checkpoint)
    return model_dict[model_name](**kwargs)


def get_sampler(dataset, meta_df, sample_frac=1.0):
    """
    Returns the sampler that can be passed to the data loader.
    This sampler can be used to reduce the problem of class imbalance.

    Args:
        dataset (torch.utils.data.Dataset): Dataset to be used.
        meta_df (pandas.DataFrame): Meta dataframe containing the labels.
        sample_frac (float): By what fraction should the data be sampled. Default: 1.0

    returns:
        sampler (torch.utils.data.sampler.WeightedRandomSampler): Sampler to be used
    """
    if "category" not in meta_df.columns:
        raise ValueError("category column not present in meta_df")

    weights = dict(
        zip(range(len(dataset)), (1 / meta_df["category"].value_counts()).values)
    )
    sample_weights = [weights[item[-1]] for item in dataset]
    sampler = WeightedRandomSampler(
        sample_weights, int(len(sample_weights) * sample_frac), replacement=True
    )
    return sampler


def main():
    args = ArgParser()

    config = configparser.ConfigParser()
    config.read(args.config)

    BATCH_SIZE = args.batch_size
    MAX_EPOCH = args.max_epochs
    llogger = CSVLogger("emotions", args.log_dir)

    data_path = config["data"]["audio"]
    cat_df = get_meta(config["data"]["label"], "category")
    dial_df = get_meta(config["data"]["text"], "dial")
    meta_df = (
        pd.merge(cat_df, dial_df, on="file")
        .groupby(["file", "dial"])
        .head(1)
        .reset_index()
        .drop(columns=["index"])
    )
    meta_df["category"] = meta_df["category"].str.replace(";.*", "", regex=True)
    
    labels = [
            'happiness', 
            'sadness', 
            'neutral state', 
            'anger',  
            'frustration', 
            # 'fear'
            ]
    meta_df = meta_df[meta_df['category'].isin(labels)]
    meta_df['category'] =  meta_df["category"].map(dict(zip(labels,range(len(labels)))))
    meta_df.reset_index(drop = True, inplace = True)

    meta_df.to_csv('meta.csv', index = False)
    meta_df = pd.read_csv('sampled.csv')
    checkpoint_file = None
    if args.resume_from_checkpoint:
        checkpoint_path = os.path.join('emotions', args.log_dir, f"version_{args.resume_from_checkpoint}", 'checkpoints')
        checkpoint_file = os.path.join(checkpoint_path, os.listdir(checkpoint_path)[0])
    model = get_model(
        args.model, num_classes=len(meta_df["category"].unique()), from_checkpoint = checkpoint_file, **config["model"]
    )
    dataset = AudioDataset(meta_df, data_path, modality=args.model)
    train_set, val_set = torch.utils.data.random_split(
        dataset,
        [0.8, 0.2],
    )
    #sampler = get_sampler(train_set, meta_df, sample_frac=args.sample_frac)


    train_dataloader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        num_workers=args.num_workers,
        drop_last=True,
        #shuffle=True,
        #sampler=sampler,
    )
    checkpoint_callback = ModelCheckpoint(monitor="val_f1")

    test_dataloader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        #sampler = test_sampler,
        num_workers=args.num_workers,
        drop_last=True,
    )
    
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCH,
        #gradient_clip_val=1,
        logger=llogger,
        callbacks = [checkpoint_callback]
    )

    trainer.fit(
        model,
        train_dataloader,
        test_dataloader,
    )


if __name__ == "__main__":
    main()
