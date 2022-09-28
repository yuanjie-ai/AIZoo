#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : DeepNN.
# @File         : Add
# @Time         : 2020/4/10 11:20 下午
# @Author       : yuanjie
# @Email        : yuanjie@xiaomi.com
# @Software     : PyCharm
# @Description  : 
import tensorflow as tf


class Add(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        # Be sure to call this somewhere!
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        if not isinstance(inputs, list):
            return inputs
        if len(inputs) == 1:
            return inputs[0]
        if len(inputs) == 0:
            return tf.constant([[0.0]])

        return tf.keras.layers.add(inputs) # 为了通用
