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
from torch.utils.data import TensorDataset, DataLoader

import pytorch_lightning as pl

from sklearn.model_selection import train_test_split as _train_test_split, StratifiedKFold, ShuffleSplit

# ME
from meutils.pipe import *
from aizoo.common import check_classification

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class TorchModule(pl.LightningModule):

    # def __init__(self, *args: Any, **kwargs: Any) -> None:
    #     super().__init__(*args, **kwargs)

    def predict4dataloader(self, dataloader: DataLoader, batch_hook=None):
        """

        @param dataloader:
        @param batch_hook: 校验训练集与测试集，比如会多个 label，由 forward 决定
        @return:
        """

        for batch in tqdm(dataloader):
            if batch_hook:
                batch = batch_hook(batch)
            yield self.predict_proba(*batch)

    def predict(self, *arrays):
        return self.predict_proba(*arrays).argmax(-1)

    @torch.no_grad()
    def predict_proba(self, *arrays):  # 可以优化 model(torch.tensor(input_ids[:1]), torch.tensor(attention_masks[:1]))

        if self.training:
            self.train(False)

        return self(*array2tensor(arrays)).detach().softmax(-1)

    def fit(self, max_epochs=1,
            train_dataloaders: DataLoader = None,
            val_dataloaders: DataLoader = None,
            gpus=0,
            fast_dev_run=True,  # debug
            callbacks=None,
            ckpt_path=None,
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

        if trainer_kwargs is None:
            trainer_kwargs = {}

        trainer = pl.Trainer(
            max_epochs=max_epochs, gpus=gpus, fast_dev_run=fast_dev_run,
            callbacks=callbacks,
            **trainer_kwargs
        )
        trainer.fit(self, train_dataloaders, val_dataloaders, ckpt_path=ckpt_path)
        return self


class TorchData(object):
    """OOF
    df = GoldenDataset('携程酒店评论').dataframe.dropna().reset_index(drop=1)

    data = tokenizer(df.review.tolist(), max_length=MAX_LENGTH, truncation=True, pad_to_max_length=True)
    data['label'] = df.label

    dff = pd.DataFrame(dict(data))

    for train_index, valid_index in StratifiedKFold(n_splits=5, shuffle=True).split(dff, dff.label):
        df_train, df_valid = dff.iloc[train_index], dff.iloc[valid_index]

        train_dataloader = TorchData().from_cache(df_train['input_ids'], df_train['attention_mask'], df_train['label'])
        valid_dataloader = TorchData().from_cache(df_valid['input_ids'], df_valid['attention_mask'], df_valid['label'])

        break

    """

    def __init__(self, batch_size=128):
        self.batch_size = batch_size

    def train_test_split(self, *arrays, test_size=0.2, random_state=42, stratify=None):
        """
            data = tokenizer(df.review.tolist(), max_length=MAX_LENGTH, truncation=True, pad_to_max_length=True)
            data['label'] = df.label
            train_dataloader, test_dataloader = TorchData(batch_size=128).train_test_split(
                data['input_ids'],
                data['attention_mask'],
                data['label']
                )
        @param arrays:
        @param test_size:
        @param random_state:
        @param stratify:
        @return:
        """
        # todo支持时间序列数据
        _ = _train_test_split(*array2tensor(arrays),
                              test_size=test_size,
                              random_state=random_state,
                              stratify=stratify)

        return self.from_cache(*_[::2]), self.from_cache(*_[1::2], is_train=False)

    def from_cache(self, *inputs, is_train=True, device='cpu'):
        """出入参长度一致

            X = np.random.random((1000, 2))
            y = X @ (2, 1) + 1
            ds = Data(batch_size=5).from_cache(X, y)
        """
        # 输入
        # logger.info(f"The {'train' if is_train else ' test'}'s shape: {inputs[0].shape}")

        # combine featues and labels of dataset

        dataset = TensorDataset(*array2tensor(inputs, device=device))

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


def array2tensor(arrays, device='cpu') -> List[torch.Tensor]:
    data = []
    for a in arrays:
        assert isinstance(a, (list, pd.Series, np.ndarray, pd.DataFrame, torch.Tensor)), "`arrays` Data Type Error"

        if isinstance(a, (list, np.ndarray, torch.Tensor)):
            pass

        elif isinstance(a, (pd.Series, pd.DataFrame)):
            a = a.values.tolist()

        data.append(torch.tensor(a, device=device))

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


if __name__ == '__main__':

    from sklearn.datasets import make_classification

    X, y = make_classification(20, n_features=5)

    for train_dataloader, test_dataloader in tqdm(TorchData(8).train_test_split(X, y)):
        for i in train_dataloader:
            print(i)
        break
