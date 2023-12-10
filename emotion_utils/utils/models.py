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
from torch.optim.lr_scheduler import ReduceLROnPlateau
class EmotionModel(pl.LightningModule):
    """
    Class for the emotion recognition model.
    This class is used as parent class
    """

    def __init__(self, num_classes=5, **kwargs):
        """
        Initializes the model

        Args:
            num_classes (int): Number of classes to be predicted. Default: 10
        """
        super(EmotionModel, self).__init__()
        self.num_classes = num_classes
        self.kwargs = kwargs
        self.dropout_rate = float(kwargs.get("dropout", 0.5))
        self.lr = float(kwargs.get("lr", 5e-4))
        self.val_f1 = torchmetrics.classification.F1Score(
                task="multiclass", num_classes=num_classes,average= 'weighted'
                )
        self.train_f1 = torchmetrics.classification.F1Score(
                task="multiclass", num_classes=num_classes,average= 'weighted'
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
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, verbose=True)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor':'val_loss'}


class AudioSpectrogramModel(EmotionModel):
    def __init__(
            self,
            num_classes=5,
            **kwargs,
            ):
        super(AudioSpectrogramModel, self).__init__()
        self.num_classes = num_classes
        self.sp_model = ASTForAudioClassification.from_pretrained(
                "ast-finetuned-audioset-10-10-0.4593", return_dict=False
                )
        for param in self.sp_model.parameters():
            param.requires_grad = False
            
        #for param in self.sp_model.audio_spectrogram_transformer.encoder.layer[-1].output.dense.parameters():
        #  param.requires_grad = True
          
        self.sp_model.classifier.dense = nn.Linear(768, 512)
        self.normalize = nn.LayerNorm(512)
        self.fc1 = nn.Linear(512, self.num_classes)
        self.normalize_fc1 = nn.LayerNorm(128)
        self.fc2 = nn.Linear(128, self.num_classes)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.relu = nn.LeakyReLU()

    def forward(
            self,
            x,
            ):

        
        x = x["input_values"].view(x["input_values"].size(0), 1024, 128)
        #x = torch.concat([x,x], dim = 1)
        out = self.sp_model(x, return_dict=False)[0]  # 256
        out = self.normalize(out)
        #out = self.dropout(out)
        out = self.fc1(out)

        return out

    def training_step(self, train_batch, batch_idx):
        signal, labels = train_batch
        labels = labels.float().to("cuda:0")
        outputs = self(signal).to("cuda:0").softmax(dim = -1)  # .argmax(1).float()
        #outputs = outputs.logit().argmax(dim = 1)
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
        #outputs = outputs.logit().argmax(dim = 1)
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
            num_classes=5,
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

        #for param in self.bert_model.encoder.layer[-1:].parameters():
        #    param.requires_grad = True

        self.sp_fc = nn.Linear(256, 128)
        self.normalize_sp = nn.LayerNorm(256)
        self.normalize_sp_fc = nn.LayerNorm(128)
        self.normalize_bert = nn.LayerNorm(128)
        self.bert_fc = nn.Linear(768, 128)
        self.fc1 = nn.Linear(256, self.num_classes)
        #self.fc2 = nn.Linear(128, self.num_classes)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.dropout_bert = nn.Dropout(min(0.9,self.dropout_rate*1.2))
        self.relu = nn.LeakyReLU()

    def forward(self, x, y):
        
        input_bert = y["input_ids"].view(y["input_ids"].size(0), 50 )
        atten_bert = y["attention_mask"].view(y["attention_mask"].size(0),50)

        x = x["input_values"].view(x["input_values"].size(0), 1024, 128)
        rounded_tensor = torch.round(x * 10 ** 4) / (10 ** 4)
        attention_mask_sp = torch.ne(rounded_tensor, 0.4670).float()
        sp = self.sp_model(x,  return_dict=False)[0]  # 256
        sp = self.normalize_sp(sp)
        sp = self.normalize_sp_fc(self.relu(self.sp_fc(sp)))

        bert, pool = self.bert_model(
                input_ids=input_bert, attention_mask=atten_bert, return_dict=False
                )
        bert = self.bert_fc(pool)
        bert = self.relu(bert)
        bert = self.normalize_bert(bert)
        out = torch.concat([bert, sp], axis=1)

        out = self.fc1(out)
        return out

    def training_step(self, train_batch, batch_idx):
        signal, dial, labels = train_batch
        labels = labels.float().to("cuda:0")
        outputs = self(signal, dial).to("cuda:0")#.softmax(dim=-1)  # .argmax(1).float()
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

        outputs = self(signal, dial).to("cuda:0")#.softmax(dim = -1)  # .argmax(1).float()
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
            num_classes=5,
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
        input_bert = y["input_ids"].view(y["input_ids"].size(0), 50 )
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
        self.train_f1(outputs, labels.int())
        self.log("train_loss", loss, on_step=False, on_epoch=True)
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

class AudioSpectrogramModel1(EmotionModel):
    def __init__(
            self,
            num_classes=5,
            **kwargs,
            ):
        super(AudioSpectrogramModel1, self).__init__()
        self.num_classes = num_classes
            
        #for param in self.sp_model.audio_spectrogram_transformer.encoder.layer[-1].output.dense.parameters():
        #  param.requires_grad = True
          
        self.conv2d = nn.Conv2d(1, 10, (5,5), )
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d((3,3))
        self.fc1 = nn.Linear(26880, 128)
        #self.fc1 = nn.Linear(131072,  64)
        self.norm = nn.LayerNorm(128)
        self.conv2=nn.Conv2d(10,20,  (5,5))
        self.maxpool2 = nn.MaxPool2d((3,3))
        self.fc2 = nn.Linear(128, self.num_classes) 
        self.dropout = nn.Dropout(self.dropout_rate)


    def forward(
            self,
            x,
            ):


        x = x["input_values"]#.view(x["input_values"].size(0), 1024, 128)
        #x = torch.concat([x,x], dim = 1)
        out = self.conv2d(x)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.maxpool2(out)

        out = torch.flatten(out, 1)
        #out = self.relu(out)

        out = self.fc1(out)
        #out = self.dropout(out)
        out =self.relu(out)
        out = self.norm(out)

        out = self.fc2(out)
        
        return out

    def training_step(self, train_batch, batch_idx):
        signal, labels = train_batch
        labels = labels.float().to("cuda:0")
        outputs = self(signal).to("cuda:0")  # .argmax(1).float()
        #outputs = outputs.logit().argmax(dim = 1)
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

        outputs = self(signal).to("cuda:0")# .argmax(1).float()
        #outputs = outputs.logit().argmax(dim = 1)
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(outputs, labels.long())
        self.val_auc(outputs, labels.int())
        self.val_f1(outputs, labels.int())
        
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log("val_auc", self.val_auc, on_step=False, on_epoch=True)
        self.log("val_f1", self.val_auc, on_step=False, on_epoch=True)

class AudioLangModelPre(EmotionModel):
    def __init__(
            self,
            num_classes=5,
            **kwargs,
            ):
        super(AudioLangModelPre, self).__init__()
        self.num_classes = num_classes
        self.sp_model = AudioSpectrogramModel()        
        self.sp_model.load_from_checkpoint("/home/usd.local/siddhi.bajracharya/jupyter/emotions/emotions/audio/version_6/checkpoints/epoch=0-step=277.ckpt")
        self.bert_model = LangModel()
        self.bert_model.load_from_checkpoint("/home/usd.local/siddhi.bajracharya/jupyter/emotions/emotions/text/version_3/checkpoints/epoch=1-step=554.ckpt")

        for param in self.sp_model.parameters():
            param.requires_grad = False
        self.sp_model.fc1= nn.Linear(512, 128)

        for param in self.bert_model.parameters():
            param.requires_grad = False

        for param in self.bert_model.parameters():
            param.requires_grad = False
        self.bert_model.fc1= nn.Linear(768, 128)




        self.fc1 = nn.Linear(256, self.num_classes)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.dropout_bert = nn.Dropout(min(0.9,self.dropout_rate*1.2))
        self.relu = nn.LeakyReLU()

    def forward(self, x, y):
        sp = self.sp_model(x,  )#[0]  # 256
        bert = self.bert_model(y)
        out = torch.concat([bert, sp], axis=1)
        out = self.fc1(out)
        return out

    def training_step(self, train_batch, batch_idx):
        signal, dial, labels = train_batch
        #         signal = signal['input_values'].view(signal['input_values'].size(0), 1024,128).shape
        labels = labels.float().to("cuda:0")
        outputs = self(signal, dial).to("cuda:0")#.softmax(dim=-1)  # .argmax(1).float()
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

        outputs = self(signal, dial).to("cuda:0")#.softmax(dim = -1)  # .argmax(1).float()
        criterion = torch.nn.CrossEntropyLoss()

        loss = criterion(outputs, labels.long())
        self.val_auc(outputs, labels.int())
        self.val_f1(outputs, labels.int())

        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log("val_auc", self.val_auc, on_step=False, on_epoch=True)
        self.log("val_f1", self.val_f1, on_step=False, on_epoch=True)


