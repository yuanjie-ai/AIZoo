#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : Python.
# @File         : DNN
# @Time         : 2020-03-13 13:42
# @Author       : yuanjie
# @Email        : yuanjie@xiaomi.com
# @Software     : PyCharm
# @Description  :
from typing import Any, Callable, Dict, List, Optional, Sequence, Type, Union

import tensorflow as tf

class DNN(tf.keras.layers.Layer):

    def __init__(self,
                 hidden_units_list: List[int] = (64, 32, 16, 4),
                 activation='relu',
                 kernel_regularizer=tf.keras.regularizers.l2(),
                 use_bn=False,
                 dropout_rate=0,
                 seed=666,
                 name='DNN',
                 **kwargs):
        super().__init__(name=name, **kwargs)

        self.hidden_units_list = hidden_units_list
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.kernel_regularizer = kernel_regularizer
        self.use_bn = use_bn
        self.seed = seed
        self.num_layer = len(self.hidden_units_list)

    def build(self, input_shape):
        super().build(input_shape)  # self.built = True

        self.dense_layers = []
        for index, units in enumerate(self.hidden_units_list):
            _ = tf.keras.layers.Dense(
                units,
                activation=self.activation,
                use_bias=True,
                kernel_initializer='glorot_uniform',
                bias_initializer='zeros',
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=None,
                name=f"dense{index}"
            )
            self.dense_layers.append(_)

        # BN
        if self.use_bn:
            self.bn_layers = [tf.keras.layers.BatchNormalization() for _ in range(self.num_layer)]

        self.dropout_layers = []
        for i in range(self.num_layer):
            _ = tf.keras.layers.Dropout(self.dropout_rate, seed=self.seed + i)
            self.dropout_layers.append(_)

    def call(self, inputs, training=None, **kwargs):
        """http://www.luyixian.cn/news_show_256709.aspx
        BN和Dropout共同使用时会出现的问题
        BN和Dropout单独使用都能减少过拟合并加速训练速度，但如果一起使用的话并不会产生1+1>2的效果，相反可能会得到比单独使用更差的效果。
        相关的研究参考论文：Understanding the Disharmony between Dropout and Batch Normalization by Variance Shift
        本论文作者发现理解 Dropout 与 BN 之间冲突的关键是网络状态切换过程中存在神经方差的（neural variance）不一致行为。
        试想若有图一中的神经响应 X，当网络从训练转为测试时，Dropout 可以通过其随机失活保留率（即 p）来缩放响应，并在学习中改变神经元的方差，
        而 BN 仍然维持 X 的统计滑动方差。这种方差不匹配可能导致数值不稳定（见下图中的红色曲线）。而随着网络越来越深，最终预测的数值偏差可能会累计，
        从而降低系统的性能。简单起见，作者们将这一现象命名为「方差偏移」。
        事实上，如果没有 Dropout，那么实际前馈中的神经元方差将与 BN 所累计的滑动方差非常接近（见下图中的蓝色曲线），这也保证了其较高的测试准确率。
        """
        deep_input = inputs
        for i in range(self.num_layer):
            fc = self.dense_layers[i](deep_input)  # 注意下次循环的输入
            fc = self.bn_layers[i](fc, training=training) if self.use_bn else fc
            fc = self.dropout_layers[i](fc, training=training)
            deep_input = fc

        return deep_input

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.hidden_units_list[-1],)

    def get_config(self):
        base_config = super().get_config()
        config = {
            'hidden_units': self.hidden_units_list,
            'activation': self.activation,
            'kernel_regularizer': self.kernel_regularizer,
            'use_bn': self.use_bn,
            'dropout_rate': self.dropout_rate,
            'seed': self.seed
        }
        return {**base_config, **config}
