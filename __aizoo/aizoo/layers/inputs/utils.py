#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : DeepNN.
# @File         : utils
# @Time         : 2020/4/13 6:49 下午
# @Author       : yuanjie
# @Email        : yuanjie@xiaomi.com
# @Software     : PyCharm
# @Description  :
from collections import OrderedDict
from enum import Enum


# from collections import namedtuple
# FeatType = namedtuple('FeatType', ['Cat', 'Num'])('Cat', 'Num')
class FeatureType(str, Enum):
    DenseFeature = "DenseFeature"
    SparseFeature = "SparseFeature"
    SequenceFeature = "SequenceFeature"


################################################################

import tensorflow as tf
from tensorflow.python.feature_column.feature_column_v2 import \
    FeatureColumn, NumericColumn, CategoricalColumn, SequenceCategoricalColumn, EmbeddingColumn, \
    BucketizedColumn

from typing import Any, Callable, Dict, List, Optional, Sequence, Type, Union
from tensorflow.keras.layers import Input


# tf.feature_column.numeric_column
# from tensorflow.feature_column import sequence_categorical_column_with_identity


def feature_columns_to_feature_layer_inputs(feature_columns: List[FeatureColumn], prefix=''):
    feature_layer_inputs = OrderedDict()

    for fc in feature_columns:
        if isinstance(fc, NumericColumn):
            feature_layer_inputs[fc.name] = Input(shape=fc.shape, name=prefix + fc.name, dtype=fc.dtype)
        elif isinstance(fc, CategoricalColumn):
            feature_layer_inputs[fc.name] = Input(shape=(1,), name=prefix + fc.name, dtype=fc.dtype)

        elif isinstance(fc, SequenceCategoricalColumn):  # todo: 序列特征、带权序列特征
            feature_layer_inputs[fc.name] = Input(shape=(fc.maxlen,), name=prefix + fc.name, dtype=fc.dtype)
            if fc.weight_name is not None:
                feature_layer_inputs[fc.weight_name] = Input(shape=(fc.maxlen, 1), name=prefix + fc.weight_name,
                                                             dtype="float32")
            if fc.length_name is not None:
                feature_layer_inputs[fc.length_name] = Input((1,), name=prefix + fc.length_name, dtype='int32')


        else:
            raise TypeError("Invalid feature column type,got", type(fc))

    return feature_layer_inputs
