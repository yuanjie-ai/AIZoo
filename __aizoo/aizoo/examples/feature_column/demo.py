#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : DeepNN.
# @File         : demo
# @Time         : 2020/4/13 10:28 上午
# @Author       : yuanjie
# @Email        : yuanjie@xiaomi.com
# @Software     : PyCharm
# @Description  : https://blog.csdn.net/u014061630/article/details/82937333


import numpy as np
import pandas as pd

import tensorflow as tf

from tensorflow import feature_column
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

URL = 'https://storage.googleapis.com/applied-dl/heart.csv'
dataframe = pd.read_csv(URL)
dataframe.head()

train, test = train_test_split(dataframe, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)
print(len(train), 'train examples')
print(len(val), 'validation examples')
print(len(test), 'test examples')


# A utility method to create a tf.data dataset from a Pandas Dataframe
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    dataframe = dataframe.copy()
    labels = dataframe.pop('target')
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds


batch_size = 5  # A small batch sized is used for demonstration purposes
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

age = feature_column.numeric_column("age")

feature_columns = []
feature_layer_inputs = {}
# feature_column.crossed_column
# numeric cols
for header in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'slope', 'ca']:
    feature_columns.append(feature_column.numeric_column(header))
    feature_layer_inputs[header] = tf.keras.Input(shape=(1,), name=header)

# bucketized cols
age_buckets = feature_column.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
feature_columns.append(age_buckets)

# indicator cols
thal = feature_column.categorical_column_with_vocabulary_list(
    'thal', ['fixed', 'normal', 'reversible'])
thal_one_hot = feature_column.indicator_column(thal)
feature_columns.append(thal_one_hot)
feature_layer_inputs['thal'] = tf.keras.Input(shape=(1,), name='thal', dtype=tf.string)

# embedding cols
thal_embedding = feature_column.embedding_column(thal, dimension=8)
feature_columns.append(thal_embedding)

# crossed cols
crossed_feature = feature_column.crossed_column([age_buckets, thal], hash_bucket_size=1000)
crossed_feature = feature_column.indicator_column(crossed_feature)
feature_columns.append(crossed_feature)

batch_size = 32
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

# 网络
feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
feature_layer_outputs = feature_layer(feature_layer_inputs)

x = layers.Dense(128, activation='relu')(feature_layer_outputs)
x = layers.Dense(64, activation='relu')(x)

baggage_pred = layers.Dense(1, activation='sigmoid')(x)

model = keras.Model(inputs=feature_layer_inputs, outputs=baggage_pred)

# 获取thal_embedding权重
model_ = keras.Model(inputs=feature_layer_inputs['thal'],
                     outputs=tf.keras.layers.DenseFeatures([thal_embedding])({'thal': feature_layer_inputs['thal']}))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(train_ds)

# 传字典也行
feed_dict = train_ds.as_numpy_iterator().__next__()
model.fit(*feed_dict)

# # estimator
# print("Estimator")
#
#
# def input_fn(features, labels, training=True, batch_size=256):
#     """An input function for training or evaluating"""
#     # 将输入转换为数据集。
#     dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
#
#     # 如果在训练模式下混淆并重复数据。
#     if training:
#         dataset = dataset.shuffle(1000).repeat()
#
#     return dataset.batch(batch_size)
#
#
# m = tf.estimator.DNNClassifier(
#     feature_columns=feature_columns,
#     hidden_units=[64, 32],
#     #     model_dir='./model_log',
#     n_classes=2
# )
# m.train(input_fn=lambda: input_fn(dataframe.drop('target', 1), dataframe.target), steps=5000)


#
# from tensorflow.feature_column import numeric_column
#

from tensorflow.python.feature_column.feature_column_v2 import NumericColumn, CategoricalColumn




