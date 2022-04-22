#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : aizoo.
# @File         : utils
# @Time         : 2021/9/15 上午10:11
# @Author       : yuanjie
# @WeChat       : 313303303
# @Software     : PyCharm
# @Description  :

import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.model_selection import train_test_split as _train_test_split

# ME
from meutils.pipe import *

callbacks = [
    ModelCheckpoint(monitor='val_loss', mode='min', verbose=True),
    EarlyStopping(monitor='val_loss', mode='min', verbose=True)
]


class TorchModule(pl.LightningModule):
    """
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

    # def __init__(self, *args: Any, **kwargs: Any) -> None:
    #     super().__init__(*args, **kwargs)

    def predict(self, *arrays):
        return self.predict_proba(*arrays).argmax(-1)

    def predict_proba(self, *arrays):  # 可以优化 model(torch.tensor(input_ids[:1]), torch.tensor(attention_masks[:1]))
        # batch_size = 1024
        # _ = TorchData(batch_size).from_cache(*arrays, False)
        # return self.eval()(_).detach().softmax(-1)
        if self.training:
            self.train(False)

        return self(*array2tensor(arrays)).detach().softmax(-1)

    @classmethod
    def fit(cls, max_epochs=1,
            train_dataloaders: DataLoader = None,
            val_dataloaders: DataLoader = None,
            trainer_kwargs=None,
            seed=42,
            *args: Any, **kwargs: Any):
        """https://blog.csdn.net/qq_27135095/article/details/122654805

            callbacks = [ModelCheckpoint, EarlyStopping] # todo 设计通用的callbacks

        @param max_epochs:
        @param trainer_kwargs:
        @param args:
        @param kwargs:
        @return:
        """
        pl.seed_everything(seed)
        model = cls(*args, **kwargs)

        if trainer_kwargs is None:
            trainer_kwargs = {}  # torch.cuda.is_available()

        trainer = pl.Trainer(max_epochs=max_epochs, **trainer_kwargs)
        trainer.fit(model, train_dataloaders, val_dataloaders)
        return model


class TorchData(object):

    def __init__(self, batch_size=128, cache_filename=""):
        self.batch_size = batch_size
        self.cache_filename = cache_filename

    def _cv(self):  # todo: StratifiedKFold
        pass

    def train_test_split(self, *arrays, test_size=0.2, random_state=42, stratify=None):
        f"""{_train_test_split.__doc__}"""

        # todo支持时间序列数据
        _ = _train_test_split(*array2tensor(arrays),
                              test_size=test_size,
                              random_state=random_state,
                              stratify=stratify)

        return self.from_cache(*_[::2]), self.from_cache(*_[1::2], is_train=False)

    def from_cache(self, *inputs, is_train=True):
        """出入参长度一致

            X = np.random.random((1000, 2))
            y = X @ (2, 1) + 1
            ds = Data(batch_size=5).from_cache(X, y)
        """
        # 输入
        logger.info(f"The {'train' if is_train else ' test'}'s shape: {inputs[0].shape}")

        # combine featues and labels of dataset

        dataset = TensorDataset(*array2tensor(inputs))

        # put dataset into DataLoader
        dataloader = DataLoader(
            dataset=dataset,  # torch TensorDataset format
            batch_size=self.batch_size,  # mini batch size
            shuffle=is_train,  # whether shuffle the data or not
            num_workers=0,  # read data in multithreading
        )

        return dataloader

    def save(self, ds, filename):
        """torch.save / joblib.dump"""
        return torch.save(ds, filename)

    def load(self, filename):
        return torch.load(filename)


def array2tensor(arrays) -> List[torch.Tensor]:
    data = []
    for a in arrays:
        assert isinstance(a, (list, pd.Series, np.ndarray, pd.DataFrame, torch.Tensor)), "`arrays` Data Type Error"

        if isinstance(a, torch.Tensor):
            pass

        elif isinstance(a, (list, pd.Series, np.ndarray)):
            a = torch.tensor(a)

        elif isinstance(a, pd.DataFrame):
            a = a.values.tolist()

        data.append(a)

    return data


def define_device(device_name):
    """
    Define the device to use during training and inference.
    If auto it will detect automatically whether to use cuda or cpu

    Parameters
    ----------
    device_name : str
        Either "auto", "cpu" or "cuda"

    Returns
    -------
    str
        Either "cpu" or "cuda"
    """
    if device_name == "auto":
        if torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    elif device_name == "cuda" and not torch.cuda.is_available():
        return "cpu"
    else:
        return device_name
