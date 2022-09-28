#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : inn.
# @File         : DeepFM
# @Time         : 2020/5/22 10:03 上午
# @Author       : yuanjie
# @Email        : yuanjie@xiaomi.com
# @Software     : PyCharm
# @Description  : https://img-blog.csdn.net/2018050820141665?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NvbmdiaW54dQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70
from typing import Any, Callable, Dict, List, Optional, Sequence, Type, Union

from inn.models.BaseModel import BaseModel

import tensorflow as tf
from inn.features.fc import *

from inn.layers.DNN import DNN
from inn.layers.interaction import FM
from inn.features.fc.utils import get_linear_output

from inn.layers.EmbeddingPooling import EmbeddingPooling


class DeepFM(BaseModel):

    def __init__(self, fcs: List[Column],
                 fm_layer=FM(),
                 dnn_layer=DNN(),
                 num_class=2,
                 model_name='DeepFM',
                 **kwargs):
        super().__init__(fcs=fcs,
                         num_class=num_class,
                         model_name=model_name,
                         **kwargs)
        self.fm_layer = fm_layer
        self.dnn_layer = dnn_layer

    def _build_model(self, **kwargs):
        """
        sigmoid(concatenate(fm, dnn))
        线性部分：多数值dense(1) + 每个cat embedding(1)
        二阶：
        FM部分：FM(embedding(k))

        """
        nc_outs = []
        cc_outs = []
        for fc, input in self.fc2input:
            if isinstance(fc, NumericColumn):
                _ = tf.keras.layers.Dense(1, activation=None)(input)  # activation
                nc_outs.append(_)

            elif isinstance(fc, (CategoricalColumn, SequenceCategoricalColumn)):
                _ = tf.keras.layers.Embedding(fc.vocabulary_size, 1)(input)
                _1 = tf.keras.layers.GlobalAveragePooling1D()(_)

                _2 = tf.keras.layers.Embedding(fc.vocabulary_size, fc.embedding_dim)(input)

                cc_outs.append((_1, _2))

        linear_out = cc_outs + [i for i, j in cc_outs]
        if linear_out:
            if len(linear_out) > 1:
                linear_out = tf.keras.layers.concatenate(nc_outs)
            else:
                linear_out = linear_out[0]

        if cc_outs:
            if len(cc_outs) > 1:
                _ = tf.keras.layers.concatenate([j for i, j in cc_outs], axis=1)
                fm_out = self.fm_layer(_)

                _ = tf.keras.layers.concatenate([tf.keras.layers.GlobalAveragePooling1D()(j) for i, j in cc_outs])

            else:
                fm_out = self.fm_layer(cc_outs[0])







        dnn_out = self.dnn(k_out)
        dnn_out = tf.keras.layers.Dense(1, use_bias=False, activation=None)(dnn_out)

        # 合并输出
        out = tf.keras.layers.concatenate([fm_out, dnn_out])  # 为啥是＋
        out = self.prediction_output_layer(out)

        model = tf.keras.models.Model(inputs=self.inputs, outputs=out)
        return model


if __name__ == '__main__':
    fcs = [NumericColumn('num'), CategoricalColumn('cat')]

    print(DeepFM(fcs).summary())
