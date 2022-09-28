#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : inn.
# @File         : DeepFM
# @Time         : 2020/5/22 10:03 上午
# @Author       : yuanjie
# @Email        : yuanjie@xiaomi.com
# @Software     : PyCharm
# @Description  : https://img-blog.csdn.net/2018050820141665?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NvbmdiaW54dQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70
from typing import Any, Callable, Dict, List, Optional, Sequence, Type, Union

from tensorflow.python.feature_column.feature_column_v2 import FeatureColumn

from inn.models.BaseModel import BaseModel

import tensorflow as tf
from inn.features.fc import *

from inn.layers.DNN import DNN
from inn.layers.interaction import FM

numeric_column = tf.feature_column.numeric_column
categorical_column_with_identity = tf.feature_column.categorical_column_with_identity

nums = ['num1', 'num2']
cats = ['cat1', 'cat2']
multi_cats = ['mcat1', 'mcat2']

feature_columns = []
ncs = [numeric_column(i) for i in nums]
ccs = [categorical_column_with_identity(i, 10) for i in cats]


def fc2input(fcs: List[FeatureColumn]):
    for fc in fcs:
        yield tf.keras.layers.Input(shape=fc.shape, name=fc.name, dtype=fc.dtype)

list(fc2input(ncs))
