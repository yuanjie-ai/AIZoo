#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : inn.
# @File         : Multihot
# @Time         : 2020/5/20 10:42 上午
# @Author       : yuanjie
# @Email        : yuanjie@xiaomi.com
# @Software     : PyCharm
# @Description  : 

from sklearn.feature_extraction.text import CountVectorizer


class MultiHot(CountVectorizer):

    def __init__(self, **kwargs):
        super().__init__(tokenizer=lambda x: x, lowercase=False, **kwargs)


import tensorflow as tf
from tensorflow.python.feature_column import feature_column_lib


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

    # _ = tf.keras.layers.DenseFeatures([column])({'feature': array})
    return _
