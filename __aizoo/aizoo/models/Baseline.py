#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : inn.
# @File         : DNN
# @Time         : 2020/5/19 10:55 上午
# @Author       : yuanjie
# @Email        : yuanjie@xiaomi.com
# @Software     : PyCharm
# @Description  : 

from typing import Any, Callable, Dict, List, Optional, Sequence, Type, Union

import tensorflow as tf
from inn.models import BaseModel
from inn.layers import DNN, Prediction
from inn.layers.inputs.utils import feature_columns_to_feature_layer_inputs

from tensorflow.python.feature_column.feature_column_v2 import \
    FeatureColumn, NumericColumn, CategoricalColumn, SequenceCategoricalColumn, EmbeddingColumn, \
    BucketizedColumn


class Baseline(BaseModel):

    def __init__(self, feature_columns: List[FeatureColumn], **kwargs):
        super().__init__(feature_columns=feature_columns, **kwargs)  # 覆盖基础属性

    @property
    def model(self):
        feature_layer_inputs = feature_columns_to_feature_layer_inputs(self.feature_columns)
        feature_layer_outputs = tf.keras.layers.DenseFeatures(self.feature_columns)(feature_layer_inputs)

        x = DNN()(feature_layer_outputs)

        output = Prediction(task=self.task, num_class=self.num_class)(x)

        model = tf.keras.models.Model(inputs=feature_layer_inputs, outputs=output)
        model.summary()
        return model
