#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : DeepTricks.
# @File         : __init__.py
# @Time         : 2019-09-10 15:21
# @Author       : yuanjie
# @Email        : yuanjie@xiaomi.com
# @Software     : PyCharm
# @Description  : 

from .Dice import Dice

import tensorflow as tf
from tensorflow.keras.layers import Activation
from tensorflow.keras.utils import get_custom_objects


class Mish(Activation):
    '''
    Mish Activation Function.
    .. math::
        mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^{x}))
    Shape:
        - Input: Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
        - Output: Same shape as the input.
    Examples:
        >>> X = Activation('Mish', name="conv1_act")(X_input)
    '''

    def __init__(self, activation, **kwargs):
        super(Mish, self).__init__(activation, **kwargs)
        self.__name__ = 'Mish'


def mish(inputs):
    return inputs * tf.math.tanh(tf.math.softplus(inputs))


get_custom_objects().update({'Mish': Mish(mish)})


def custom_gelu(x):
    return 0.5 * x * (1 + tf.tanh(tf.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))


get_custom_objects().update({'custom_gelu': Activation(custom_gelu)})

# 注册函数
funcs = [mish, custom_gelu]
custom_objects = {func.__name__: Activation(func) for func in funcs}

get_custom_objects().update(custom_objects)
