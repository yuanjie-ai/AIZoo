#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : DeepNN.
# @File         : _layers
# @Time         : 2020/4/10 11:58 下午
# @Author       : yuanjie
# @Email        : yuanjie@xiaomi.com
# @Software     : PyCharm
# @Description  :

import tensorflow as tf


class Linear(tf.keras.layers.Layer):
    def __init__(self, units=32, **kwargs):
        super(Linear, self).__init__(**kwargs)
        self.units = units

    # build方法一般定义Layer需要被训练的参数。
    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(shape=(self.units,),
                                 initializer='random_normal',
                                 trainable=True)
        # self.built = True
        super(Linear, self).build(input_shape)  # 相当于设置self.built = True

    # call方法一般定义正向传播运算逻辑，__call__方法调用了它。
    def call(self, inputs, **kwargs):
        return tf.matmul(inputs, self.w) + self.b

    # 如果要让自定义的Layer通过Functional API 组合成模型时可以序列化，需要自定义get_config方法。
    def get_config(self):
        config = super(Linear, self).get_config()
        config.update({'units': self.units})
        return config

    @classmethod
    def test_compute_output_shape(cls, input_shape=(None, 16)):
        """若推断失败，需重写`compute_output_shape`"""
        layer = cls()
        layer.build(input_shape)
        _ = layer.compute_output_shape(input_shape)
        print(_)

        # 如果built = False，调用__call__时会先调用build方法, 再调用call方法。
        # layer(tf.random.uniform((66, input_shape[-1])))


if __name__ == '__main__':
    print(Linear.test_compute_output_shape((100,)))