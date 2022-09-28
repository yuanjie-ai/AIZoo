#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : inn.
# @File         : feature_column
# @Time         : 2020/5/20 10:38 上午
# @Author       : yuanjie
# @Email        : yuanjie@xiaomi.com
# @Software     : PyCharm
# @Description  : 

from typing import Any, Callable, Dict, List, Optional, Sequence, Type, Union

from collections import OrderedDict

import tensorflow as tf
from tensorflow.keras.layers import Input

from tensorflow.python.feature_column.feature_column_v2 import \
    FeatureColumn, NumericColumn, \
    CategoricalColumn, SequenceCategoricalColumn, BucketizedColumn, \
    IndicatorColumn, EmbeddingColumn, IdentityCategoricalColumn


def feature_columns_to_feature_layer_inputs(feature_columns: List[FeatureColumn]):
    feature_layer_inputs = OrderedDict()

    for fc in feature_columns:
        # NumericColumn: BucketizedColumn 离散化
        if isinstance(fc, NumericColumn):
            feature_layer_inputs[fc.name] = Input(shape=fc.shape, name=fc.name, dtype=fc.dtype)

        # CategoricalColumn: IndicatorColumn + EmbeddingColumn
        elif isinstance(fc, IndicatorColumn):

            if isinstance(fc.categorical_column, IdentityCategoricalColumn):
                dtype = tf.int64
            else:
                dtype = fc.categorical_column.dtype

            feature_layer_inputs[fc.name[:-10]] = Input(shape=(1,), name=fc.name[:-10], dtype=dtype)


        elif isinstance(fc, CategoricalColumn):
            feature_layer_inputs[fc.name] = Input(shape=(1,), name=fc.name, dtype=fc.dtype)

        elif isinstance(fc, SequenceCategoricalColumn):  # todo: 序列特征、带权序列特征
            feature_layer_inputs[fc.name] = Input(shape=(fc.maxlen,), name=fc.name, dtype=fc.dtype)
            if fc.weight_name is not None:
                feature_layer_inputs[fc.weight_name] = Input(shape=(fc.maxlen, 1), name=fc.weight_name,
                                                             dtype="float32")
            if fc.length_name is not None:
                feature_layer_inputs[fc.length_name] = Input((1,), name=fc.length_name, dtype='int32')


        elif isinstance(fc, (BucketizedColumn, EmbeddingColumn)):  # 有些特征列是建立在其他输入之上的，不需要输入
            pass

        else:
            raise TypeError("Invalid feature column type,got", type(fc))

    return feature_layer_inputs


# EmbeddingColumn
def embedding_column(feature_columns: List[FeatureColumn], get_dim=lambda nunique: int(6 * nunique ** 0.25)):
    ecs = []
    for fc in feature_columns:
        assert isinstance(fc,
                          CategoricalColumn), f"{fc} object has no attribute 'num_buckets'"  # hasattr(fc, 'num_buckets')

        dim = get_dim(fc.num_buckets + 1) if callable(get_dim) else get_dim  # oov
        ec = tf.feature_column.embedding_column(fc, dim)
        ecs.append(ec)

    return ecs

#################################

tf.feature_column.numeric_column
tf.feature_column.categorical_column_with_identity
tf.feature_column.categorical_column_with_hash_bucket
