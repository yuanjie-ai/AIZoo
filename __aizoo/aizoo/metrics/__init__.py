#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : DeepTricks.
# @File         : __init__.py
# @Time         : 2019-09-11 12:02
# @Author       : yuanjie
# @Email        : yuanjie@xiaomi.com
# @Software     : PyCharm
# @Description  : 

import tensorflow as tf
from tensorflow.keras.utils import get_custom_objects

from sklearn.metrics import roc_auc_score


# # 注册函数
# funcs = [tf.keras.metrics.AUC()]
# custom_objects = {func.__name__: Activation(func) for func in funcs}
#
# get_custom_objects().update()