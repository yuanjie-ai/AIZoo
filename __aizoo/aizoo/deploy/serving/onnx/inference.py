#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : Python.
# @File         : inference
# @Time         : 2020-03-16 23:13
# @Author       : yuanjie
# @Email        : yuanjie@xiaomi.com
# @Software     : PyCharm
# @Description  : 
import numpy as np
import onnxruntime as rt


class Inference(object):

    def __init__(self, path="./iris.onnx"):
        self.sess = rt.InferenceSession(path)
        self._describe()

    def run(self, X):
        _ = self.sess.get_inputs()[0]
        self.input_name = _.name
        # self.input_shape = tuple(_.shape)
        # assert X.shape == self.input_shape

        if not isinstance(X[0][0], np.float32):
            X = X.astype(np.float32)
        return self.sess.run(None, {self.input_name: X})  # 概率输出 or 类别输出

    def _describe(self):

        for attr_ in ['get_inputs', 'get_outputs']:
            _puts = self.sess.__getattribute__(attr_)()
            for i, _put in enumerate(_puts, 1):
                print({attr_.split('_')[-1][:-1].title() + str(i): (_put.type, _put.name, _put.shape)})
