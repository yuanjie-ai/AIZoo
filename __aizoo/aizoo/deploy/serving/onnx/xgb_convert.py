#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : Python.
# @File         : xgb_convert
# @Time         : 2020-03-16 13:15
# @Author       : yuanjie
# @Email        : yuanjie@xiaomi.com
# @Software     : PyCharm
# @Description  :
# https://github.com/amesar/mlflow-examples/blob/220e03d6457ac9a2ac08589a96881691679678a3/scala/onnx/README.md
# https://github.com/amesar/mlflow-examples/tree/220e03d6457ac9a2ac08589a96881691679678a3
# https://github.com/onnx/tutorials/blob/master/tutorials/PytorchTensorflowMnist.ipynb


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)

import xgboost as xgb

xgb_model = xgb.XGBClassifier()
xgb_model = xgb_model.fit(X_train, y_train)

print("Test data accuracy of the xgb classifier is {:.2f}".format(xgb_model.score(X_test, y_test)))

from onnxmltools.convert import convert_xgboost, convert_lightgbm
from onnxconverter_common.data_types import FloatTensorType

onnx_model = convert_xgboost(xgb_model, initial_types=[("input", FloatTensorType([1, 4]))])

with open("gbtree.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

if __name__ == '__main__':
    from inference import Inference

    infer = Inference("gbtree.onnx")
    print(infer.run(X[:1]))
