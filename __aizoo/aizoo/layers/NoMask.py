#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : DeepNN.
# @File         : NoMask
# @Time         : 2020/4/10 11:33 下午
# @Author       : yuanjie
# @Email        : yuanjie@xiaomi.com
# @Software     : PyCharm
# @Description  : 
import tensorflow as tf


class NoMask(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(NoMask, self).__init__(**kwargs)

    def build(self, input_shape):
        # Be sure to call this somewhere!
        super(NoMask, self).build(input_shape)

    def call(self, x, mask=None, **kwargs):
        return x

    def compute_mask(self, inputs, mask=None):
        return None
