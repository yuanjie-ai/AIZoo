#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : Python.
# @File         : torch_convert
# @Time         : 2020-03-16 15:57
# @Author       : yuanjie
# @Email        : yuanjie@xiaomi.com
# @Software     : PyCharm
# @Description  : 

import numpy as np
import torch
from torch import nn, onnx
import torch.nn.functional as F
from sklearn.datasets import load_iris

iris = load_iris()
X, y = iris.data, iris.target


class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 16)
        self.fc2 = nn.Linear(16, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # F.softmax
        return x


net = NN()
print(net)

# Export: X.shape
onnx.export(
    net, torch.randn(128, 4), './iris.onnx', verbose=True,
    input_names=['input_name'],
    output_names=['output_name']
)

if __name__ == '__main__':
    from inference import Inference

    infer = Inference()
    print(infer.run(X[:128])[:5])
