#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : inn.
# @File         : Prediction
# @Time         : 2020/5/19 6:14 下午
# @Author       : yuanjie
# @Email        : yuanjie@xiaomi.com
# @Software     : PyCharm
# @Description  : 

import tensorflow as tf
from .utils import get_activation_by_num_class


class Prediction(tf.keras.layers.Layer):

    def __init__(self, num_class=2, name='Prediction', **kwargs):
        super().__init__(name=name, **kwargs)

        self.num_class = num_class
        self.activation = get_activation_by_num_class(num_class)

    def build(self, input_shape):
        super().build(input_shape)
        units = self.num_class if self.num_class > 2 else 1  # 多分类输出 num_class 维
        self.fc = tf.keras.layers.Dense(units, activation=self.activation)

    def call(self, inputs, **kwargs):
        return self.fc(inputs)

    def compute_output_shape(self, input_shape):
        if self.task == "multiclass":
            return (None, self.num_class)
        else:
            return (None, 1)

    def get_config(self, ):
        base_config = super().get_config()
        config = {'task': self.task, 'num_class': self.num_class}
        return {**base_config, **config}
