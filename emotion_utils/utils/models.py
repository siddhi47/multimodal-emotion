"""
    author:siddhi47
    desc: Contains the model for the emotion recognition (using text, audio and both)
"""

import torch
import torchmetrics
from torch import nn
import torch.optim as optim
import lightning.pytorch as pl
from transformers import BertModel
from torchaudio.transforms import *
from emotion_utils.utils.utils import *
from transformers import ASTForAudioClassification

class EmotionModel(pl.LightningModule):
    """
    Class for the emotion recognition model.
    This class is used as parent class
    """

    def __init__(self, num_classes=10, **kwargs):
        """
        Initializes the model

        Args:
            num_classes (int): Number of classes to be predicted. Default: 10
        """
        super(EmotionModel, self).__init__()
        self.num_classes = num_classes
        self.kwargs = kwargs
        self.dropout_rate = float(kwargs.get("dropout", 0.5))
        self.lr = float(kwargs.get("lr", 1e-3))

        self.val_f1 = torchmetrics.classification.F1Score(
            task="multiclass", num_classes=num_classes
        )
        self.train_f1 = torchmetrics.classification.F1Score(
            task="multiclass", num_classes=num_classes
        )

        self.train_auc = torchmetrics.classification.AUROC(
            task="multiclass", num_classes=num_classes
        )
        self.val_auc = torchmetrics.classification.AUROC(
            task="multiclass", num_classes=num_classes
        )

    def configure_optimizers(self):
        """
        Configures the optimizer to be used

        returns:
            optimizer (torch.optim.Optimizer): Optimizer to be used
        """
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


class AudioSpectrogramModel(EmotionModel):
    def __init__(
        self,
        num_classes=10,
        **kwargs,
    ):
        super(AudioSpectrogramModel, self).__init__()
        self.num_classes = num_classes
        self.sp_model = ASTForAudioClassification.from_pretrained(
            "ast-finetuned-audioset-10-10-0.4593", return_dict=False
        )
        for param in self.sp_model.parameters():
            param.requires_grad = False
        self.sp_model.classifier.dense = nn.Linear(768, 527)

        self.fc1 = nn.Linear(527, self.num_classes)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.relu = nn.ReLU()

    def forward(
        self,
        x,
    ):
        x = x["input_values"].view(x["input_values"].size(0), 1024, 128)

        out = self.sp_model(x, return_dict=False)[0]  # 256
        out = self.dropout(out)
        out = self.fc1(out)
        #         out = self.sigmoid(out)
        return out

    def training_step(self, train_batch, batch_idx):
        signal, labels = train_batch
        labels = labels.float().to("cuda:0")
        outputs = self(signal).to("cuda:0")  # .argmax(1).float()
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(outputs, labels.long())
        self.train_auc(outputs, labels.int())
        self.train_f1(outputs, labels.int())

        self.log("loss", loss, on_step=False, on_epoch=True)
        self.log("train_auc", self.train_auc, on_step=False, on_epoch=True)
        self.log("train_f1", self.train_auc, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, val_batch, batch_idx):
        signal, labels = val_batch
        labels = labels.float().to("cuda:0")
        #         signal = signal['input_values'].view(signal['input_values'].size(0), 1024,128).shape

        outputs = self(signal).to("cuda:0")  # .argmax(1).float()
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(outputs, labels.long())
        self.val_auc(outputs, labels.int())

        self.val_f1(outputs, labels.int())
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log("val_auc", self.val_auc, on_step=False, on_epoch=True)
        self.log("val_f1", self.val_auc, on_step=False, on_epoch=True)


class AudioLangModel(EmotionModel):
    def __init__(
        self,
        num_classes=10,
        **kwargs,
    ):
        super(AudioLangModel, self).__init__()
        self.num_classes = num_classes
        self.sp_model = ASTForAudioClassification.from_pretrained(
            "ast-finetuned-audioset-10-10-0.4593", return_dict=False
        )

        self.bert_model = BertModel.from_pretrained("bert-base-uncased")
        for param in self.sp_model.parameters():
            param.requires_grad = False
        self.sp_model.classifier.dense = nn.Linear(768, 256)

        for param in self.bert_model.parameters():
            param.requires_grad = False

        for param in self.bert_model.encoder.layer[-1:].parameters():
            param.requires_grad = True

        self.fc1 = nn.Linear(1024, 256)
        self.fc2 = nn.Linear(256, self.num_classes)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.relu = nn.ReLU()

    def forward(self, x, y):
        x = x["input_values"].view(x["input_values"].size(0), 1024, 128)
        input_bert = y["input_ids"].view(y["input_ids"].size(0), 50)
        atten_bert = y["attention_mask"].view(y["attention_mask"].size(0), 50)
        #         y = y['input_ids']
        #         print(y.shape)
        sp = self.sp_model(x, return_dict=False)[0]  # 256
        bert, pool = self.bert_model(
            input_ids=input_bert, attention_mask=atten_bert, return_dict=False
        )
        bert = self.dropout(pool)
        sp = self.dropout(sp)

        out = torch.concat([bert, sp], axis=1)
        out = self.dropout(out)
        out = self.dropout(self.fc1(out))
        out = self.fc2(out)
        

        return out

    def training_step(self, train_batch, batch_idx):
        signal, dial, labels = train_batch
        #         signal = signal['input_values'].view(signal['input_values'].size(0), 1024,128).shape
        labels = labels.float().to("cuda:0")
        outputs = self(signal, dial).to("cuda:0")  # .argmax(1).float()
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(outputs, labels.long())
        self.train_auc(outputs, labels.int())
        self.train_f1(outputs, labels.int())

        self.log("loss", loss, on_step=False, on_epoch=True)
        self.log("train_auc", self.train_auc, on_step=False, on_epoch=True)
        self.log("train_f1", self.train_f1, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, val_batch, batch_idx):
        signal, dial, labels = val_batch
        labels = labels.float().to("cuda:0")
        #         signal = signal['input_values'].view(signal['input_values'].size(0), 1024,128).shape

        outputs = self(signal, dial).to("cuda:0")  # .argmax(1).float()
        criterion = torch.nn.CrossEntropyLoss()

        loss = criterion(outputs, labels.long())
        self.val_auc(outputs, labels.int())
        self.val_f1(outputs, labels.int())

        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log("val_auc", self.val_auc, on_step=False, on_epoch=True)
        self.log("val_f1", self.val_f1, on_step=False, on_epoch=True)


class LangModel(EmotionModel):
    def __init__(
        self,
        num_classes=10,
        **kwargs,
    ):
        super(LangModel, self).__init__()
        self.bert_model = BertModel.from_pretrained("bert-base-uncased")
        self.num_classes = num_classes

        for param in self.bert_model.parameters():
            param.requires_grad = False

        self.fc1 = nn.Linear(768, self.num_classes)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.relu = nn.ReLU()

    def forward(self, y):
        input_bert = y["input_ids"].view(y["input_ids"].size(0), 50)
        atten_bert = y["attention_mask"].view(y["attention_mask"].size(0), 50)
        bert, pool = self.bert_model(
            input_ids=input_bert, attention_mask=atten_bert, return_dict=False
        )
        bert = self.dropout(pool)
        out = self.fc1(bert)

        return out

    def training_step(self, train_batch, batch_idx):
        dial, labels = train_batch
        labels = labels.float().to("cuda:0")
        outputs = self(dial).to("cuda:0")  # .argmax(1).float()
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(outputs, labels.long())
        self.train_auc(outputs, labels.int())
        self.log("train_auc", self.train_auc, on_step=False, on_epoch=True)
        self.log("train_f1", self.train_f1, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        dial, labels = val_batch
        labels = labels.float().to("cuda:0")
        outputs = self(dial).to("cuda:0")  # .argmax(1).float()
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(outputs, labels.long())
        self.val_auc(outputs, labels.int())
        self.val_f1(outputs, labels.int())
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log("val_auc", self.val_auc, on_step=False, on_epoch=True)

class FaceModel(EmotionModel):
    def __init__(
            self,
            num_classes = 10, 
            embedding_dim = 256,
            **kwargs
            ):
        super(FaceModel, self).__init__()
        self.num_classes = num_classes
        self.embs = nn.Embedding(embedding_dim, 128)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=128, nhead=16), num_layers=6)
    
    def forward(self,x):
        out = self.transformer_encoder(x)
        out = out.mean(1)
        out = self.fc1(out)
        return out

    def training_step(self, train_batch, batch_idx):
        dial, labels = train_batch
        labels = labels.float().to("cuda:0")
        outputs = self(dial).to("cuda:0")  # .argmax(1).float()
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(outputs, labels.long())
        self.train_auc(outputs, labels.int())
        self.log("train_auc", self.train_auc, on_step=False, on_epoch=True)
        self.log("train_f1", self.train_f1, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        dial, labels = val_batch
        labels = labels.float().to("cuda:0")
        outputs = self(dial).to("cuda:0")  # .argmax(1).float()
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(outputs, labels.long())
        self.val_auc(outputs, labels.int())
        self.val_f1(outputs, labels.int())
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log("val_auc", self.val_auc, on_step=False, on_epoch=True)

