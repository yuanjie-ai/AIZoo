#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : inn.
# @File         : EmbeddingPooling
# @Time         : 2020/5/22 4:12 下午
# @Author       : yuanjie
# @Email        : yuanjie@xiaomi.com
# @Software     : PyCharm
# @Description  : 

import tensorflow as tf

K = tf.keras.backend


class EmbeddingPooling(tf.keras.layers.Layer):
    def __init__(self, input_dim,
                 output_dim,
                 combiner=None,
                 mask_zero=True,
                 input_length=None,
                 name='EmbeddingPooling',
                 **kwargs):
        super().__init__(name=name, **kwargs)

        self.input_dim = input_dim
        self.output_dim = output_dim  # output_dim==1 要压平
        self.combiner = combiner
        self.mask_zero = mask_zero
        self.input_length = input_length

    def build(self, input_shape):
        super().build(input_shape)  # self.built = True

        # embeddings_initializer = 'uniform',
        # embeddings_regularizer = None,
        # activity_regularizer = None,

        self.embedding_layer = tf.keras.layers.Embedding(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            mask_zero=self.mask_zero,
            input_length=self.input_length)

        if self.combiner:
            if self.combiner == 'max':
                self.pool_layer = tf.keras.layers.GlobalMaxPooling1D()

            elif self.combiner == 'mean':
                self.pool_layer = tf.keras.layers.GlobalAveragePooling1D()

            else:
                raise ValueError(f"Unsupported combiner {self.combiner}")

    def call(self, inputs, **kwargs):
        _ = self.embedding_layer(inputs)
        if self.combiner:
            return self.pool_layer(_)
        else:
            return _

    def compute_output_shape(self, input_shape):
        if self.combiner:
            return (None, self.output_dim)
        else:
            return (None, self.input_dim, self.output_dim)

    def get_config(self, ):
        base_config = self.get_config()
        config = {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "combiner": self.combiner,
            "mask_zero": self.mask_zero,
            "input_length": self.input_length
        }
        return {**base_config, **config}
