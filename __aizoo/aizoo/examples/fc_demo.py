#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : MeUtils.
# @File         : fc_demo
# @Time         : 2021/4/29 5:53 下午
# @Author       : yuanjie
# @WeChat       : 313303303
# @Software     : PyCharm
# @Description  : 

import tensorflow as tf
from tensorflow.keras.layers import DenseFeatures, Input
from tensorflow import feature_column as fc
from tensorflow.python.feature_column.feature_column_v2 import FeatureColumn
from tensorflow.python.feature_column import feature_column_lib

# tf.compat.v1.feature_column.linear_model
# me
from meutils.pipe import *


# _ = tf.keras.layers.DenseFeatures([column])({'feature': array})

class FeatureColumnName(BaseConfig):
    numeric_column: List[str] = None
    categorical_column: List[str] = None
    categorical_column_with_hash_bucket: List[str] = None

    sequence_numeric_column: List[str] = None
    sequence_categorical_column: List[str] = None


# 特征名分组
# numeric_columns
feature_layer_inputs = OrderedDict()

fcn = FeatureColumnName.parse_obj({'numeric_column': ['a', 'b', 'c', 'd']})

# {k: v for k, v in fcn.dict().items() if v}


names = fcn.numeric_column
for name in names:
    feature_layer_inputs[name] = Input(shape=fc.shape, name=prefix + fc.name, dtype=fc.dtype)

fc.numeric_column

from tensorflow.python.feature_column.feature_column_v2 import \
    FeatureColumn, NumericColumn, CategoricalColumn, SequenceCategoricalColumn, EmbeddingColumn, \
    BucketizedColumn


def fc2input(fcs: List[FeatureColumn], prefix=''):
    inputs = OrderedDict()

    for fc in fcs:
        if isinstance(fc, NumericColumn):
            inputs[fc.name] = Input(shape=fc.shape, name=prefix + fc.name, dtype=fc.dtype)
        elif isinstance(fc, CategoricalColumn):
            inputs[fc.name] = Input(shape=(1,), name=prefix + fc.name, dtype=fc.dtype)

        elif isinstance(fc, SequenceCategoricalColumn):  # todo: 序列特征、带权序列特征
            inputs[fc.name] = Input(shape=(fc.maxlen,), name=prefix + fc.name, dtype=fc.dtype)
            if fc.weight_name is not None:
                inputs[fc.weight_name] = Input(shape=(fc.maxlen, 1), name=prefix + fc.weight_name,
                                               dtype="float32")
            if fc.length_name is not None:
                inputs[fc.length_name] = Input((1,), name=prefix + fc.length_name, dtype='int32')

        else:
            raise TypeError("Invalid feature column type,got", type(fc))

    return feature_layer_inputs
