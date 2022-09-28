#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : DeepNN.
# @File         : examples
# @Time         : 2020/4/20 8:21 下午
# @Author       : yuanjie
# @Email        : yuanjie@xiaomi.com
# @Software     : PyCharm
# @Description  : https://mp.weixin.qq.com/s/5jfOahNKnUjTre0O2655IA


import tensorflow as tf
from tensorflow.python.feature_column import feature_column_lib
# tf.sparse.to_dense()



def categorical_list_column():
    column = tf.feature_column.categorical_column_with_vocabulary_list(
        # 特征列的名称
        key="feature",
        # 有效取值列表，列表的下标对应转换的数值。即，value1会被映射为0
        vocabulary_list=["value1", "value2", "value3"],
        # 取值的类型，只支持string和integer，这个会根据vocabulary_list自动推断出来
        dtype=tf.string,
        # 当取值不在vocabulary_list中时，会被映射的数值，默认为-1
        # 当该值不为-1时，num_oov_buckets必须设置为0。即两者不能同时起作用
        default_value=-1,
        # 作用同default_value，但是两者不能同时起作用。
        # 将超出的取值映射到[len(vocabulary), len(vocabulary) + num_oov_buckets)内
        # 默认取值为0
        # 当该值不为0时，default_value必须设置为-1
        # 当default_value和num_oov_buckets都取默认值时，会被映射为-1
        num_oov_buckets=3)
    feature_cache = feature_column_lib.FeatureTransformationCache(features={
        # feature对应的值可以为Tensor，也可以为SparseTensor
        "feature": tf.constant(value=[
            [["value1", "value2"], ["value3", "value3"]],
            [["value3", "value5"], ["value4", "value4"]]
        ])
    })
    # IdWeightPair(id_tensor, weight_tensor)
    return column.get_sparse_tensors(transformation_cache=feature_cache, state_manager=None)


if __name__ == '__main__':
    c = categorical_list_column()
    print(c.id_tensor.dtype)
    print(c.id_tensor.shape)
    print(c.id_tensor.dense_shape)
    print(c.id_tensor.indices)
