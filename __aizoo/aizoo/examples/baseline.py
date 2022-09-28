#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : MeUtils.
# @File         : baseline
# @Time         : 2021/4/23 1:38 下午
# @Author       : yuanjie
# @WeChat       : 313303303
# @Software     : PyCharm
# @Description  :

from meutils.pipe import *

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

from sklearn.datasets import make_classification

tf.get_logger().setLevel(40)  # logging.ERROR
K.set_learning_phase(True)

if tf.__version__ >= '2.0.0':
    tf.compat.v1.disable_eager_execution()

X, y = make_classification(10000, n_features=5, shift=0.1)

def create_model():
    """from tensorflow.python.keras.models import clone_and_build_model"""
    model = Sequential()
    model.add(Dense(12, input_dim=X.shape[1], kernel_initializer="uniform", activation="relu"))
    model.add(Dense(8, kernel_initializer="uniform", activation="relu"))
    model.add(Dense(1, kernel_initializer="uniform", activation="sigmoid"))
    return model



model = create_model()
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])


model.fit(X, y)

model.save('baseline_model/1', save_format='tf')