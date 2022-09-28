#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : aizoo.
# @File         : albert_classification
# @Time         : 2022/4/26 下午4:21
# @Author       : yuanjie
# @WeChat       : meutils
# @Software     : PyCharm
# @Description  :

import torch
import torch.nn as nn

from transformers import AutoModel, AutoTokenizer, AdamW, AutoModelForSequenceClassification
from transformers import get_linear_schedule_with_warmup

from torchmetrics.functional import accuracy

# ME
from meutils.pipe import *
from minidata import GoldenDataset, MODEL_HOME
from aizoo.torch_utils import TorchData, TorchModule

# 常量
RANDOM_SEED = 42
MAX_EPOCHS = 1
BATCH_SIZE = 128
MAX_LENGTH = 128
PRE_TRAINED_MODEL_NAME = MODEL_HOME / 'ckiplab/albert-tiny-chinese'

df = GoldenDataset('携程酒店评论').dataframe.dropna().reset_index(drop=1)

tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
clf = AutoModelForSequenceClassification.from_pretrained(PRE_TRAINED_MODEL_NAME, num_labels=2)


class PL(TorchModule):

    def __init__(self, loss_fn=nn.CrossEntropyLoss()):
        super().__init__()
        self.loss_fn = loss_fn
        # Define PyTorch model
        self.model = clf

    def forward(self, input_ids, attention_mask):
        _ = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).logits
        # loss, logits = model(input_ids, attention_mask=attention_mask, labels=labels)
        return _

    def common_step(self, batch):
        y = batch[2]
        logits = self(batch[0], batch[1])
        return logits, y

    def training_step(self, batch, batch_idx):
        logits, y = self.common_step(batch)
        loss = self.loss_fn(logits, y)
        return loss

    def validation_step(self, batch, batch_idx):
        logits, y = self.common_step(batch)
        loss = self.loss_fn(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def predicts_step(self, batch, batch_idx, dataloader_idx=0):
        print(batch_idx, dataloader_idx)
        return self(batch[:2])

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=2e-5, correct_bias=False)
        lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=1000)

        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}


"""
   from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

   callbacks = [
       ModelCheckpoint(monitor='val_loss', mode='min', verbose=True),
       EarlyStopping(monitor='val_loss', mode='min', verbose=True)
   ]
   # 重写configure_optimizers()函数即可
   # 设置优化器
   def configure_optimizers(self):
       weight_decay = 1e-6  # l2正则化系数
       # 假如有两个网络，一个encoder一个decoder
       optimizer = optim.Adam([{'encoder_params': self.encoder.parameters()}, {'decoder_params': self.decoder.parameters()}], lr=learning_rate, weight_decay=weight_decay)
       # 同样，如果只有一个网络结构，就可以更直接了
       optimizer = optim.Adam(my_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
       # 我这里设置2000个epoch后学习率变为原来的0.5，之后不再改变
       StepLR = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2000], gamma=0.5)
       optim_dict = {'optimizer': optimizer, 'lr_scheduler': StepLR}
       return optim_dict
"""

if __name__ == '__main__':
    data = tokenizer(df.review.tolist(), max_length=MAX_LENGTH, truncation=True, pad_to_max_length=True)
    data['label'] = df.label
    train_dataloader, test_dataloader = TorchData(batch_size=128).train_test_split(
        data['input_ids'],
        data['attention_mask'],
        data['label']
    )

    pl_model = PL()
    trainer = pl_model.fit(1, train_dataloader, test_dataloader)

    trainer.predict(dataloaders=test_dataloader)

    pl_model.predict_dataloader()
