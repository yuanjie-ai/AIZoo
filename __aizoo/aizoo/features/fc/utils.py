#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : inn.
# @File         : fc
# @Time         : 2020/5/21 4:42 下午
# @Author       : yuanjie
# @Email        : yuanjie@xiaomi.com
# @Software     : PyCharm
# @Description  : 

from . import *

from typing import Any, Callable, Dict, List, Optional, Sequence, Type, Union

import tensorflow as tf
from tensorflow.python.feature_column.feature_column_v2 import \
    FeatureColumn, NumericColumn, CategoricalColumn, SequenceCategoricalColumn, EmbeddingColumn, \
    BucketizedColumn


def dense_feature_out(fc: FeatureColumn, data: dict):
    return tf.keras.layers.DenseFeatures([fc])(data)


def fc2input(fc: FeatureColumn):
    return tf.keras.layers.Input(shape=fc.shape, name=fc.name, dtype=fc.dtype)


#################################

def get_linear_output(fc2input, dense_dim=1, dense_activation=None):
    dense_out = list(get_dense_out(fc2input, dense_dim=dense_dim, dense_activation=dense_activation))
    embedding_out = list(get_embedding_out(fc2input, 1))
    return tf.keras.layers.concatenate(dense_out + embedding_out)


def get_dense_out(fc2input, dense_dim=1, dense_activation=None):
    for fc, input in fc2input:
        if isinstance(fc, NumericColumn):
            _ = tf.keras.layers.Dense(dense_dim, activation=dense_activation)(input)
            yield _


def get_embedding_out(fc2input, embedding_dim=None):
    for fc, input in fc2input:
        if isinstance(fc, CategoricalColumn):
            _ = tf.keras.layers.Embedding(fc.vocabulary_size,
                                          fc.embedding_dim if embedding_dim is None else embedding_dim,
                                          mask_zero=True)(input)
            if embedding_dim == 1:
                _ = tf.keras.layers.Flatten()(_)
            yield _

        elif isinstance(fc, SequenceCategoricalColumn):
            _ = tf.keras.layers.Embedding(fc.vocabulary_size,
                                          fc.embedding_dim if embedding_dim is None else embedding_dim,
                                          mask_zero=True)(input)

            if embedding_dim == 1:
                if fc.combiner == 'mean':
                    _ = tf.keras.layers.GlobalAveragePooling1D()(_)
                    yield _

                elif fc.combiner == 'max':
                    _ = tf.keras.layers.GlobalMaxPooling1D()(_)
                    yield _
                else:
                    raise ValueError(f'Unsupported combiner {fc.combiner}')
            else:
                yield _
