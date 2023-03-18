#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : Python.
# @File         : transformer
# @Time         : 2022/10/14 上午9:01
# @Author       : yuanjie
# @WeChat       : meutils
# @Software     : PyCharm
# @Description  :

import numpy as np
from sklearn.preprocessing import FunctionTransformer


def sin_transformer(period=24):
    """可用于时间编码，比如 1-24 小时"""
    return FunctionTransformer(lambda x: np.sin(x / period * 2 * np.pi))


def cos_transformer(period=24):
    return FunctionTransformer(lambda x: np.cos(x / period * 2 * np.pi))
