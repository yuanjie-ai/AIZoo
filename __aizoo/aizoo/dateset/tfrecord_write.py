#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : DeepNN.
# @File         : tfrecord_write
# @Time         : 2020/4/22 1:29 下午
# @Author       : yuanjie
# @Email        : yuanjie@xiaomi.com
# @Software     : PyCharm
# @Description  : 


import tensorflow as tf


def _float_list(value):
    if not isinstance(value, list):
        value = [value]

    float_list = tf.train.FloatList(value=value)
    return tf.train.Feature(float_list=float_list)


def tf_train_example(feature=None):
    if feature is None:
        feature = {}
    _ = tf.train.Example(features=tf.train.Features(feature=feature))
    return _


with tf.io.TFRecordWriter('./tf.record') as f:
    features = tf.train.Features(
        feature={
            'num': _float_list(1),
            'seq': _float_list([1, 2, 3]),
            'label': _float_list(0)
        })
    eaxmple = tf.train.Example(features=features)
    f.write(eaxmple.SerializeToString())


def parser_fn(example_photo):
    features = {
        'num': tf.io.FixedLenFeature((), tf.float32),
        'seq': tf.io.FixedLenSequenceFeature((), tf.float32, allow_missing=True),
        'label': tf.io.FixedLenFeature((), tf.float32),

    }
    parsed_features = tf.io.parse_single_example(example_photo, features=features)
    return parsed_features


data = (
    tf.data.TFRecordDataset('./tf.record')
        .map(parser_fn)
)

print(list(data.as_numpy_iterator()))
