#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : DeepNN.
# @File         : mcdropout
# @Time         : 2020/5/13 7:55 下午
# @Author       : yuanjie
# @Email        : yuanjie@xiaomi.com
# @Software     : PyCharm
# @Description  : 

import numpy as np
import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
inp = tf.keras.layers.Input(shape=(28, 28))
x = tf.keras.layers.Flatten()(inp)
x = tf.keras.layers.Dense(512, activation=tf.nn.relu)(x)
x = tf.keras.layers.Dropout(0.5)(x, training=True)  # dropout 在训练和测试时都将开着
out = tf.keras.layers.Dense(10, activation=tf.nn.softmax)(x)
model = tf.keras.Model(inp, out)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=3)  # 在测试过程，dropout 也是打开的，得到的结果将会有波动，而不是完全一致

# dropout 层一直处于打开的状态，测试过程重复进行多次。
for _ in range(10):
    r = model.predict(x_test[:1])
    idx = np.argmax(r)
    print(idx, r[:, idx])
