import torch
import torchmetrics
from torch import nn
import torch.optim as optim
import lightning.pytorch as pl
from transformers import  BertModel
from torchaudio.transforms import *
from emotion_utils.utils.utils import *
from transformers import  ASTForAudioClassification


class AudioSpectrogramModel(pl.LightningModule):
    def __init__(
        self,
        num_classes=10,
    ):
        super(AudioSpectrogramModel, self).__init__()
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

        self.sp_model = ASTForAudioClassification.from_pretrained(
            "ast-finetuned-audioset-10-10-0.4593", return_dict=False
        )
        for param in self.sp_model.parameters():
            param.requires_grad = False
        self.sp_model.classifier.dense = nn.Linear(768, 527)

        self.fc1 = nn.Linear(527, num_classes)
        self.dropout = nn.Dropout(0.5)
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

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.0001, weight_decay=1e-3)
        return optimizer

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


class AudioLangModel(pl.LightningModule):
    def __init__(
        self,
        num_classes=10,
    ):
        super(AudioLangModel, self).__init__()
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

        self.sp_model = ASTForAudioClassification.from_pretrained(
            "ast-finetuned-audioset-9-10-0.4593", return_dict=False
        )
        for param in self.sp_model.parameters():
            param.requires_grad = False
        self.sp_model.classifier.dense = nn.Linear(768, 527)

        self.fc1 = nn.Linear(527, num_classes)
        self.dropout = nn.Dropout(0.5)
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

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.0001, weight_decay=1e-3)
        return optimizer

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


class LangModel(pl.LightningModule):
    def __init__(
        self,
        num_classes=10,
    ):
        super(LangModel, self).__init__()
        self.train_r2 = torchmetrics.R2Score()
        self.val_r2 = torchmetrics.R2Score()
        self.train_auc = torchmetrics.classification.AUROC(
            task="multiclass", num_classes=num_classes
        )
        self.val_auc = torchmetrics.classification.AUROC(
            task="multiclass", num_classes=num_classes
        )

        self.val_f1 = torchmetrics.classification.F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.train_f1 = torchmetrics.classification.F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        )

        self.bert_model = BertModel.from_pretrained("bert-base-uncased")

        for param in self.bert_model.parameters():
            param.requires_grad = False

        self.fc1 = nn.Linear(768, num_classes)
        self.dropout = nn.Dropout(0.7)
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

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.0001, weight_decay=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        dial, labels = train_batch
        labels = labels.float().to("cuda:0")
        outputs = self(dial).to("cuda:0")  # .argmax(1).float()
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(outputs, labels.long())
        self.train_auc(outputs, labels.int())
        self.train_f1(outputs, labels.int())
        self.log("loss", loss, on_step=False, on_epoch=True)
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
        self.log("val_f1", self.val_f1, on_step=False, on_epoch=True)


