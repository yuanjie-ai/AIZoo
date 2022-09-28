#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : inn.
# @File         : dataloader
# @Time         : 2021/1/5 11:07 上午
# @Author       : yuanjie
# @Email        : yuanjie@xiaomi.com
# @Software     : PyCharm
# @Description  : 

import torch
from torch.utils.data import TensorDataset, DataLoader

from meutils.pipe import *

X = np.random.random((1000, 2))
y = X @ (2, 1) + 1

# combine featues and labels of dataset
dataset = TensorDataset(*map(torch.from_numpy, (X, y)))

# put dataset into DataLoader
data_iter = DataLoader(
    dataset=dataset,  # torch TensorDataset format
    batch_size=128,  # mini batch size
    shuffle=True,  # whether shuffle the data or not
    num_workers=1,  # read data in multithreading
)

for x, y in data_iter:
    print(x, y)
    break
