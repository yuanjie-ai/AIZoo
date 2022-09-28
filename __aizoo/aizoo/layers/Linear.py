#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : DeepNN.
# @File         : Linear
# @Time         : 2020/4/10 11:34 下午
# @Author       : yuanjie
# @Email        : yuanjie@xiaomi.com
# @Software     : PyCharm
# @Description  : 

import tensorflow as tf


class Linear(tf.keras.layers.Layer):

    def __init__(self, l2_reg=0.0, mode=0, use_bias=False, name='Linear', **kwargs):
        super().__init__(name=name, **kwargs)

        assert mode in (0, 1, 2), ValueError("mode must be 0,1 or 2")  # raise ValueError("mode must be 0,1 or 2")

        self.l2_reg = l2_reg
        self.mode = mode
        self.use_bias = use_bias

        if self.use_bias:
            self.bias = self.add_weight(name='linear_bias',
                                        shape=(1,),
                                        initializer=tf.keras.initializers.Zeros(),
                                        trainable=True)

    def build(self, input_shape):
        super().build(input_shape)  # self.built = True

        if self.mode in (1, 2):
            shape = [int(input_shape[-1]), 1] if self.mode == 1 else [int(input_shape[1][-1]), 1]
            self.kernel = self.add_weight(
                'linear_kernel',
                shape=shape,
                initializer=tf.keras.initializers.glorot_normal(),
                regularizer=tf.keras.regularizers.l2(self.l2_reg),
                trainable=True)

    def call(self, inputs, **kwargs):
        if self.mode == 0:
            sparse_input = inputs
            linear_logit = tf.reduce_sum(sparse_input, axis=-1, keep_dims=True)
        elif self.mode == 1:
            dense_input = inputs
            fc = tf.tensordot(dense_input, self.kernel, axes=(-1, 0))
            linear_logit = fc
        else:
            sparse_input, dense_input = inputs
            fc = tf.tensordot(dense_input, self.kernel, axes=(-1, 0))
            linear_logit = tf.reduce_sum(sparse_input, axis=-1, keep_dims=False) + fc
        if self.use_bias:
            linear_logit += self.bias

        return linear_logit

    def compute_output_shape(self, input_shape):
        return (None, 1)

    def compute_mask(self, inputs, mask=None):
        return None

    def get_config(self, ):
        base_config = super(Linear, self).get_config()
        config = {'mode': self.mode, 'l2_reg': self.l2_reg, 'use_bias': self.use_bias}
        return {**base_config, **config}
