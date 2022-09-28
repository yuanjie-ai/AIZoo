#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : DeepNN.
# @File         : FM
# @Time         : 2020/4/21 1:40 下午
# @Author       : yuanjie
# @Email        : yuanjie@xiaomi.com
# @Software     : PyCharm
# @Description  : 

import tensorflow as tf

K = tf.keras.backend


class FM(tf.keras.layers.Layer):
    """Factorization Machine models pairwise (order-2) feature interactions
     without linear term and bias.

      Input shape
        - 3D tensor with shape: ``(batch_size,field_size,embedding_size)``.

      Output shape
        - 2D tensor with shape: ``(batch_size, 1)``.

      References
        - [Factorization Machines](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)
    """

    def __init__(self, name='FM', **kwargs):

        super(FM, self).__init__(name=name, **kwargs)

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError(f"Unexpected inputs dimensions {len(input_shape)}, expect to be 3 dimensions")

        super(FM, self).build(input_shape)  # 相当于 self.built = True

    def call(self, inputs, **kwargs):
        """

        :param inputs:
        fm_input = np.array([
                    [[1, 3], [2, 3]],
                    [[1, 3], [2, 3]],
                    [[1, 3], [2, 3]]
                    ], dtype=np.float32) # (3, 2, 2)
        :param kwargs:
        :return:
        """

        if K.ndim(inputs) != 3:
            raise ValueError(f"Unexpected inputs dimensions {K.ndim(inputs)}, expect to be 3 dimensions")

        fm_input = inputs  # [None, SeqLen/field_size, EmbeddingDim]

        square_of_sum = tf.pow(tf.reduce_sum(fm_input, axis=1, keepdims=True), 2)  # [None, 1, EmbeddingDim]
        sum_of_square = tf.reduce_sum(fm_input * fm_input, axis=1, keepdims=True)  # [None, 1, EmbeddingDim]
        cross_term = square_of_sum - sum_of_square  # [None, 1, EmbeddingDim]
        cross_term = 0.5 * tf.reduce_sum(cross_term, axis=2, keepdims=False)  # [None, 1]

        return cross_term

    def compute_output_shape(self, input_shape):
        return (None, 1)
