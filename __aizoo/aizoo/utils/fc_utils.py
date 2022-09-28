#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : MeUtils.
# @File         : fc_utils
# @Time         : 2021/4/29 7:27 下午
# @Author       : yuanjie
# @WeChat       : 313303303
# @Software     : PyCharm
# @Description  : 


import tensorflow as tf
from tensorflow.keras.layers import DenseFeatures, Input

from tensorflow.python.feature_column import feature_column_lib

from tensorflow.python.feature_column.feature_column_v2 import \
    FeatureColumn, NumericColumn, CategoricalColumn, SequenceCategoricalColumn, EmbeddingColumn, \
    BucketizedColumn


def debug_fc(fc: FeatureColumn, array, onehot=True):
    """
    fc = tf.feature_column.numeric_column('a')
    :return:
    https://stackoverflow.com/questions/57346191/tensorflow-pad-sequence-feature-column
    """
    name = fc.name
    if isinstance(fc, NumericColumn):
        pass

    elif isinstance(fc, CategoricalColumn):
        fc = tf.feature_column.embedding_column(fc, dimension=4)
        if onehot:
            fc = tf.feature_column.indicator_column(fc)  # onehot

    return tf.keras.layers.DenseFeatures([fc])({name: array})


def multihot(array=None, num_buckets=999, default_value=0):
    categorical_column = tf.feature_column.categorical_column_with_identity(
        key="feature",
        num_buckets=num_buckets,
        default_value=default_value)

    # multi-hot编码，出现多次的相同值，会累加
    column = tf.feature_column.indicator_column(categorical_column=categorical_column)

    if array is None:
        array = [[1, 2, 1],
                 [3, 4, 5]]
    feature_cache = feature_column_lib.FeatureTransformationCache(features={"feature": tf.constant(value=array)})
    _ = column.get_dense_tensor(transformation_cache=feature_cache, state_manager=None)

    return _
