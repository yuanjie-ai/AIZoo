#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : MeUtils.
# @File         : utils
# @Time         : 2021/3/24 2:50 下午
# @Author       : yuanjie
# @WeChat       : 313303303
# @Software     : PyCharm
# @Description  : 

from meutils.pipe import *
import tensorflow as tf

# init
if os.environ.get('eager', '0') != '0' and tf.__version__ >= '2.0.0':
    tf.compat.v1.disable_eager_execution()


def keras2tf(model_path='./inception.h5', export_path='../my_image_classifier/1'):
    """todo: tf2是否可直接保存
    """
    # The export path contains the name and the version of the model
    tf.keras.backend.set_learning_phase(0)  # Ignore dropout at inference
    model = tf.keras.models.load_model(model_path)

    # Fetch the Keras session and save the model
    # The signature definition is defined by the input and output tensors
    # And stored with the default serving key
    with tf.compat.v1.keras.backend.get_session() as sess:
        tf.compat.v1.saved_model.simple_save(
            sess,
            export_path,
            inputs={'input_image': model.input},  # 具体输入
            outputs={t.name: t for t in model.outputs}
        )
