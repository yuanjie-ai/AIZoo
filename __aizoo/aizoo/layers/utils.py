#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : DeepNN.
# @File         : utils
# @Time         : 2020/4/10 10:58 下午
# @Author       : yuanjie
# @Email        : yuanjie@xiaomi.com
# @Software     : PyCharm
# @Description  : 


import tensorflow as tf


def get_activation_by_num_class(num_class=2, task2activation=None):
    task2activation = {'regression': None, 'binary': 'sigmoid', 'multiclass': 'softmax'}
    if task2activation:
        task2activation.update(task2activation)

    if num_class == 1:
        return task2activation['regression']
    elif num_class == 2:
        return task2activation['binary']
    elif num_class >= 3:
        return task2activation['multiclass']
    else:
        raise ValueError(f"Unsupported {num_class}")


# tf.divide(x, y, name=name)
# tf.nn.softmax(logits, axis=dim, name=name)
# tf.reduce_mean(input_tensor, axis=None, keepdims=False, name=None)
# tf.reduce_sum(input_tensor, axis=None, keepdims=False, name=None)
# tf.reduce_max(input_tensor, axis=None, keepdims=False, name=None)


class Hash(tf.keras.layers.Layer):
    """
    hash the input to [0,num_buckets)
    if mask_zero = True,0 or 0.0 will be set to 0,other value will be set in range[1,num_buckets)
    """

    def __init__(self, num_buckets, mask_zero=False, **kwargs):
        self.num_buckets = num_buckets
        self.mask_zero = mask_zero
        super(Hash, self).__init__(**kwargs)

    def build(self, input_shape):
        # Be sure to call this somewhere!
        super(Hash, self).build(input_shape)

    def call(self, x, mask=None, **kwargs):
        if x.dtype != tf.string:
            x = tf.as_string(x, )

        hash_x = tf.strings.to_hash_bucket_fast(x, self.num_buckets if not self.mask_zero else self.num_buckets - 1,
                                                name=None)  # weak hash
        if self.mask_zero:
            mask_1 = tf.cast(tf.not_equal(x, "0"), 'int64')
            mask_2 = tf.cast(tf.not_equal(x, "0.0"), 'int64')
            mask = mask_1 * mask_2
            hash_x = (hash_x + 1) * mask
        return hash_x

    def compute_mask(self, inputs, mask=None):
        return None

    def get_config(self, ):
        config = {'num_buckets': self.num_buckets, 'mask_zero': self.mask_zero}
        base_config = super(Hash, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
