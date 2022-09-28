#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : DeepTricks.
# @File         : DataLoader
# @Time         : 2019-10-21 13:01
# @Author       : yuanjie
# @Email        : yuanjie@xiaomi.com
# @Software     : PyCharm
# @Description  :
"""
cache
"""
import numpy as np
import pandas as pd
import tensorflow as tf

from functools import partial


class Dataset(object):
    """
    https://tensorflow.google.cn/guide/data?hl=zh_cn

    """

    def __init__(self, batchsize=128, cache_filename=""):

        self.batchsize = batchsize
        self.cache_filename = cache_filename

    def from_cache(self, inputs, outputs=None, is_test=False, shuffle_buffer_size=10000, shuffle_seed=None):
        """
        多输入多输出inputs/outputs对应元组
        """
        # 输入
        assert isinstance(inputs, (tuple, list, np.ndarray, dict, pd.DataFrame)), "`inputs` Data Type Error"

        if isinstance(inputs, pd.DataFrame):
            inputs = inputs.to_dict('list')

        if outputs is None:
            tensors = (inputs,)
        else:
            tensors = (inputs, outputs)

        ds = tf.data.Dataset.from_tensor_slices(tensors)
        if outputs is None or is_test:  # 避免测试集shuffle
            pass
        else:
            ds = ds.shuffle(shuffle_buffer_size, seed=shuffle_seed)

        ds = ds.batch(self.batchsize)
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)  # todo: .repeat() 更乱一些？？？在每个epoch内将图片打乱组成大小为32的batch，并重复10次。

        return ds  # .repeat(epochs)

    def from_generator(self):
        # TODO: 增加对文件的操作（txt/tfrecord）
        # tf.data.Dataset.from_generator()
        pass

    def _from_np_array(self, array):
        # Common
        buffer_size = len(array)
        ds = tf.data.Dataset.from_tensor_slices(array)
        ds = ds.shuffle(buffer_size, seed=None)
        ds = ds.batch(self.batchsize)
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

        return ds  # .repeat(epochs)

    def _from_pd_dataframe(self, df: pd.DataFrame, label="label"):
        """
        import tensorflow as tf
        dataset = tf.data.Dataset.from_tensor_slices(({"a": [1, 2], "b": [3, 4]}, [0, 1]))
        print(list(dataset.as_numpy_iterator()))

        :param df:
        :param label:
        :return:
        """
        if label and label in df.columns:
            df = df.drop(labels=[label], axis=1)
            labels = df[label]
            tensors = (df.to_dict('list'), labels)  # df.to_dict('series')
        else:
            tensors = df.to_dict('list')

        # Common
        buffer_size = len(df)
        ds = tf.data.Dataset.from_tensor_slices(tensors)
        ds = ds.shuffle(buffer_size, seed=None)
        ds = ds.batch(self.batchsize)
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

        # features_dataset = tf.data.Dataset.from_tensor_slices(features)
        # labels_dataset = tf.data.Dataset.from_tensor_slices(labels)
        # tf.data.Dataset.zip((features_dataset, labels_dataset))
        return ds  # .repeat(epochs)

    def from_tfrecord(self,
                      feature_dict: dict,
                      file_pattern: str = None,
                      label='label',
                      file_shuffle=True,
                      file_shuffle_seed=666,
                      shuffle_buffer_size=0):
        """注意缺失值问题
        ds = Dataset()
        feature_dict = {
            'id': tf.io.FixedLenFeature((), tf.int64, default_value=0),
            'feature': tf.io.FixedLenFeature((), tf.int64, default_value=0) # default_value=tf.zeros([], dtype=tf.float32)
            }
        ds = ds.from_tfrecord(feature_dict, '/Users/yuanjie/Desktop/Projects/Spark/MIPush/test-output.tfrecord/part*')

        :param feature_dict:
        :param file_pattern:
        :param shuffle:
        :param seed:
        :param shuffle_buffer_size:
        :return:
        """
        # TODO: cache
        # assert isinstance(filename, str), f"file path error: {filename}"
        #
        # if Path(filename).is_dir():
        #     filename = list(map(str, Path(filename).glob(glob_regex))) # tf.data.Dataset.list_files

        # parser_fn = partial(tf.io.parse_single_example, features=feature_dict)
        parser_fn = partial(self._tfrecord_parser_fn, features=feature_dict, label=label)

        filenames = tf.data.Dataset.list_files(file_pattern, file_shuffle, file_shuffle_seed)

        # ds
        ds = tf.data.TFRecordDataset(filenames, num_parallel_reads=tf.data.experimental.AUTOTUNE)
        ds = ds.map(parser_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        if shuffle_buffer_size > 0:
            ds = ds.shuffle(shuffle_buffer_size, seed=777, reshuffle_each_iteration=True)

        ds = ds.batch(self.batchsize)
        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)  # 若内存泄露，需手动指定: TODO测试放在前后的速度
        ds = ds.interleave(
            tf.data.TFRecordDataset,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
            deterministic=False)
        return ds  # .repeat(epochs)

    def _tfrecord_parser_fn(self, example_photo, features, label='label'):
        _ = tf.io.parse_single_example(example_photo, features=features)
        return _, _.pop(label)  # X, y
