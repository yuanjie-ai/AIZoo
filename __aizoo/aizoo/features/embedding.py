#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : inn.
# @File         : embedding
# @Time         : 2020/5/21 8:16 下午
# @Author       : yuanjie
# @Email        : yuanjie@xiaomi.com
# @Software     : PyCharm
# @Description  : 

from inn.utils.seed_utils import init_seeds

init_seeds(666)

import numpy as np
import tensorflow as tf

# 单值离散型
tf.keras.layers.Input(shape=(1,))
cat_feats = np.array([1, 2])
cat_feats = np.array([[1], [2]])
_ = tf.keras.layers.Embedding(10, 4, mask_zero=True)(cat_feats)
_ = tf.keras.layers.Flatten()(_)

assert _.ndim == 2

# 多值离散型: TODO: [[[1,2], [3, 4]], [[1,2], [3, 4]]] 序列的序列
tf.keras.layers.Input(shape=(2,))

seq_cat_feats = np.array([(1, 2), (2, 1)])
_ = tf.keras.layers.Embedding(10, 4, mask_zero=True)(seq_cat_feats)
_ = tf.keras.layers.GlobalAveragePooling1D()(_)

assert _.ndim == 2

from inn.features.fc import *

# embedding outputs
fc = CategoricalColumn()


inputs = []

def get_embedding_outputs(fc=CategoricalColumn()):
    _ = columns2inputs(fc)

    if isinstance(fc, CategoricalColumn):
        return tf.keras.layers.Embedding(fc.vocabulary_size, fc.embedding_dim, mask_zero=True)(_)

    elif isinstance(fc, SequenceCategoricalColumn):
        _ = tf.keras.layers.Embedding(fc.vocabulary_size, fc.embedding_dim, mask_zero=True)(_)
        if fc.combiner == 'mean':
            return tf.keras.layers.GlobalAveragePooling1D()(_)
        elif fc.combiner == 'max':
            return tf.keras.layers.GlobalMaxPooling1D()(_)
        else:
            raise ValueError("Unknown combiner")
