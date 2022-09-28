#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : DeepNN.
# @File         : show_column
# @Time         : 2020/4/13 4:54 下午
# @Author       : yuanjie
# @Email        : yuanjie@xiaomi.com
# @Software     : PyCharm
# @Description  : https://stackoverflow.com/questions/57346191/tensorflow-pad-sequence-feature-column
import tensorflow as tf
from tensorflow.python.feature_column.feature_column import _LazyBuilder

def show_column(data: dict, feature_column):
    try:
        return show_column1(data, feature_column)
    except Exception as e:
        return show_column2(data, feature_column)


def show_column1(data: dict, feature_column):
    builder = _LazyBuilder(data)
    id_tensor, weight = feature_column._get_sparse_tensors(builder)

    if weight is None:
        return id_tensor.values
    else:
        return id_tensor.values, weight.values


def show_column2(data: dict, feature_column, feature_type='Seq'):
    """wrap a categorical column with an embedding_column or indicator_column

    :param feature_column: feature_column.numeric_column('age')
    :param batch_data: {'feature_name': [1,2,3,4]}
    :return:
    """
    if feature_type == 'Seq':
        feature_layer = tf.keras.experimental.SequenceFeatures([feature_column])
        sequence_input, sequence_length = feature_layer(data)
        return sequence_input, sequence_length
    else:
        feature_layer = tf.keras.layers.DenseFeatures([feature_column])
        return feature_layer(data).numpy()
