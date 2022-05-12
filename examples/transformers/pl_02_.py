#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : MeUtils.
# @File         : pl
# @Time         : 2021/1/4 12:00 下午
# @Author       : yuanjie
# @Email        : yuanjie@xiaomi.com
# @Software     : PyCharm
# @Description  : https://mp.weixin.qq.com/s/37MwMqZB4DfWxTnk5jsLbQ

import torch
from torch import nn

from torchmetrics.functional import accuracy

from transformers import AutoModel, AutoTokenizer, AdamW, AutoModelForSequenceClassification
from transformers import get_linear_schedule_with_warmup

# ME
from meutils.pipe import *
from minidata import MODEL_HOME
from aizoo.torch_utils import TorchData, TorchModule

# 常量
RANDOM_SEED = 42
MAX_EPOCHS = 1
BATCH_SIZE = 128
PRE_TRAINED_MODEL_NAME = MODEL_HOME / 'ckiplab/albert-tiny-chinese'

tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

# data
train = pd.read_csv('02/train.csv')
test = pd.read_csv('02/test.csv')
sub = pd.read_csv('02/sub.csv')

train['text'] = train['content'].fillna('') + '[SEP]' + train['level_1'] + '[SEP]' + train['level_2'] + '[SEP]' + train[
    'level_3'] + '[SEP]' + train['level_4']
test['text'] = test['content'].fillna('') + '[SEP]' + test['level_1'] + '[SEP]' + test['level_2'] + '[SEP]' + test[
    'level_3'] + '[SEP]' + test['level_4']

data = tokenizer.batch_encode_plus(
    train['text'].tolist(),
    max_length=128,
    truncation=True,
    pad_to_max_length=True,
    return_attention_mask=True,
    return_token_type_ids=False,
    return_tensors='np'
)

train_dataloader, test_dataloader = TorchData().train_test_split(
    data['input_ids'],
    data['attention_mask'],
    train.label.values)


class PL(TorchModule):

    def __init__(self, n_classes=2, loss_fn=nn.CrossEntropyLoss()):
        super().__init__()
        self.loss_fn = loss_fn
        # Define PyTorch model
        self.model = AutoModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.model.config.hidden_size, n_classes)  # 两个类别

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )
        output = self.drop(pooled_output)  # dropout
        return self.out(output)

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

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=2e-5, correct_bias=False)
        lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=1000)

        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}


if __name__ == '__main__':
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

    pl.seed_everything(42)

    checkpoint = ModelCheckpoint(
        monitor='val_acc',
        mode='max',
        verbose=True
    )

    early_stopping = EarlyStopping(
        monitor='val_acc',
        mode='max',
        verbose=True
    )

    pl_model = PL()
    trainer = pl.Trainer(max_epochs=30, gpus=1, callbacks=[early_stopping, checkpoint])
    trainer.fit(pl_model, train_dataloader, test_dataloader)

    # PL.fit(
    #     1, train_dataloader, test_dataloader,
    #     trainer_kwargs=dict(callbacks=[early_stopping, checkpoint])
    # )
