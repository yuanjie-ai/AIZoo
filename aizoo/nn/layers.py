#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : aizoo.
# @File         : layers
# @Time         : 2022/4/25 下午3:47
# @Author       : yuanjie
# @WeChat       : meutils
# @Software     : PyCharm
# @Description  :

from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as F

"""
BN/LN https://www.zhihu.com/question/487766088
"""


class FM(nn.Module):
    """Factorization Machine models pairwise (order-2) feature interactions
     without linear term and bias.
      Input shape
        - 3D tensor with shape: ``(batch_size,field_size,embedding_size)``.
      Output shape
        - 2D tensor with shape: ``(batch_size, 1)``.
      References
        - [Factorization Machines](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)
    """

    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        fm_input = inputs

        square_of_sum = torch.pow(torch.sum(fm_input, dim=1, keepdim=True),
                                  2)  # fm_input.sum(dim=1, keepdim=True).pow(2)
        sum_of_square = torch.sum(fm_input * fm_input, dim=1, keepdim=True)
        cross_term = square_of_sum - sum_of_square
        cross_term = 0.5 * torch.sum(cross_term, dim=2, keepdim=False)

        return cross_term


class DNN(nn.Module):
    """
    DNN(3)(torch.randn(10, 3))
    https://blog.csdn.net/kkxi123456/article/details/118313680
    """

    def __init__(self, input_size,
                 hidden_units_list: List[int] = (64, 32, 16, 4),
                 bias=True
                 ):
        super().__init__()

        self.fcs = nn.ModuleList()
        for i, hidden_size in enumerate(hidden_units_list):
            fc = nn.Linear(input_size, hidden_size, bias=bias)
            self.fcs.append(fc)

            input_size = hidden_size

    def forward(self, x):  # 开始计算的函数

        for fc in self.fcs:  # todo: add nn.BatchNorm1d nn.Dropout
            out = fc(x)
            x = out

        return x


class DNN(nn.Module):
    """The Multi Layer Percetron
      Input shape
        - nD tensor with shape: ``(batch_size, ..., input_dim)``. The most common situation would be a 2D input with shape ``(batch_size, input_dim)``.
      Output shape
        - nD tensor with shape: ``(batch_size, ..., hidden_size[-1])``. For instance, for a 2D input with shape ``(batch_size, input_dim)``, the output would have shape ``(batch_size, hidden_size[-1])``.
      Arguments
        - **inputs_dim**: input feature dimension.
        - **hidden_units**:list of positive integer, the layer number and units in each layer.
        - **activation**: Activation function to use.
        - **l2_reg**: float between 0 and 1. L2 regularizer strength applied to the kernel weights matrix.
        - **dropout_rate**: float in [0,1). Fraction of the units to dropout.
        - **use_bn**: bool. Whether use BatchNormalization before activation or not.
        - **seed**: A Python integer to use as random seed.
    """

    def __init__(self, inputs_dim, hidden_units=(64, 32, 16, 4),
                 activation='relu', l2_reg=0, dropout_rate=0, use_bn=False,
                 init_std=0.0001, dice_dim=3, seed=1024, device='cpu'):
        super().__init__()

        self.dropout = nn.Dropout(dropout_rate)
        self.seed = seed
        self.l2_reg = l2_reg
        self.use_bn = use_bn
        if len(hidden_units) == 0:
            raise ValueError("hidden_units is empty!!")
        hidden_units = [inputs_dim] + list(hidden_units)

        self.linears = nn.ModuleList(
            [nn.Linear(hidden_units[i], hidden_units[i + 1]) for i in range(len(hidden_units) - 1)])

        if self.use_bn:
            self.bn = nn.ModuleList(
                [nn.BatchNorm1d(hidden_units[i + 1]) for i in range(len(hidden_units) - 1)])

        self.activation_layers = nn.ModuleList(
            [activation_layer(activation, hidden_units[i + 1], dice_dim) for i in range(len(hidden_units) - 1)])

        for name, tensor in self.linears.named_parameters():
            if 'weight' in name:
                nn.init.normal_(tensor, mean=0, std=init_std)

        self.to(device)

    def forward(self, inputs):
        deep_input = inputs

        for i in range(len(self.linears)):

            fc = self.linears[i](deep_input)

            if self.use_bn:
                fc = self.bn[i](fc)

            fc = self.activation_layers[i](fc)

            fc = self.dropout(fc)
            deep_input = fc
        return deep_input


if __name__ == '__main__':
    DNN()
