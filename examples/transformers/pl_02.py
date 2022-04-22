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
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torchmetrics.functional import accuracy

from transformers import AutoModel, AutoTokenizer, AutoConfig, AdamW
from transformers import \
    get_linear_schedule_with_warmup, \
    get_cosine_schedule_with_warmup, \
    get_constant_schedule_with_warmup, \
    get_polynomial_decay_schedule_with_warmup

from sklearn.model_selection import train_test_split

# ME
from meutils.pipe import *

pl.seed_everything(42)

# 常量
RANDOM_SEED = 42
MAX_EPOCHS = 1
BATCH_SIZE = 128
PRE_TRAINED_MODEL_NAME = 'models/ckiplab/albert-tiny-chinese'

tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

# data
train = pd.read_csv('02/train.csv')
test = pd.read_csv('02/test.csv')
sub = pd.read_csv('02/sub.csv')

train['text'] = train['content'].fillna('') + '[SEP]' + train['level_1'] + '[SEP]' + train['level_2'] + '[SEP]' + train[
    'level_3'] + '[SEP]' + train['level_4']
test['text'] = test['content'].fillna('') + '[SEP]' + test['level_1'] + '[SEP]' + test['level_2'] + '[SEP]' + test[
    'level_3'] + '[SEP]' + test['level_4']

data = []
for _, s in train.iterrows() | xtqdm:
    text_map = tokenizer.encode_plus( # todo: tokenizer.batch_encode_plus
        s.text,
        add_special_tokens=True,
        max_length=128,
        truncation=True,  # add
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    text_map['target'] = torch.tensor(s.label)
    text_map['input_ids'] = text_map['input_ids'].flatten()
    text_map['attention_mask'] = text_map['attention_mask'].flatten()
    data.append(text_map)
    # break

X_train, X_test = train_test_split(data, test_size=0.1, random_state=RANDOM_SEED)
X_valid, X_test = train_test_split(X_test, test_size=0.5, random_state=RANDOM_SEED)


class DS(Dataset):
    """torchdata
         https://mp.weixin.qq.com/s/p2PpaxlsYDUB76jJs-59cQ
     """

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        return data

    @classmethod
    def data_loader(cls, data, batch_size=128, train=True):
        if not train:
            batch_size *= 10

        return DataLoader(cls(data), batch_size=batch_size, shuffle=train)


train_data_loader = DS.data_loader(X_train, batch_size=BATCH_SIZE)
val_data_loader = DS.data_loader(X_valid, batch_size=BATCH_SIZE, train=False)
test_data_loader = DS.data_loader(X_test, batch_size=BATCH_SIZE, train=False)


# model

class PL(pl.LightningModule):

    def __init__(self, n_classes, loss_fn=nn.CrossEntropyLoss()):
        super().__init__()
        self.loss_fn = loss_fn
        # Define PyTorch model
        self.model = AutoModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.model.config.hidden_size, n_classes)  # 类别数

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )
        output = self.drop(pooled_output)  # dropout
        return self.out(output)

    def common_step(self, batch):
        y = batch['target']
        logits = self(batch['input_ids'], batch['attention_mask']) # 与 dataloader也不能数据结构一致
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

    # def configure_optimizers(self):
    #     optimizer = AdamW(self.parameters(), lr=2e-5, correct_bias=False)
    #     return optimizer

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=2e-5, correct_bias=False)
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=1000)  # num_training_steps = len(train)*MAX_EPOCHS

        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}

    #
    # @staticmethod
    # def add_model_specific_args(parent_parser):
    #     parser = ArgumentParser(parents=[parent_parser], add_help=False)
    #     parser.add_argument('--hidden_dim', type=int, default=128)
    #     parser.add_argument('--learning_rate', type=float, default=0.0001)
    #     return parser


if __name__ == '__main__':
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
    trainer.fit(pl_model, train_data_loader, val_data_loader)

    # trainer.save_checkpoint("example.ckpt")
    # new_model = PL.load_from_checkpoint(checkpoint_path="example.ckpt")
#### DEBUG
# # 参训练集、校验集和测试集分别只加载 10%, 20%, 30%，或者使用int 型表示batch
# trainer = Trainer(
#     limit_train_batches=0.1,
#     limit_val_batches=0.2,
#     limit_test_batches=0.3
# )
#
# trainer.save_checkpoint("example.ckpt")
# new_model = PL.load_from_checkpoint(checkpoint_path="example.ckpt")
