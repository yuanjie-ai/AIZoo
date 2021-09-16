#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : aizoo.
# @File         : opt_test
# @Time         : 2021/9/16 下午1:22
# @Author       : yuanjie
# @WeChat       : 313303303
# @Software     : PyCharm
# @Description  :


from aizoo.tuner.optimizers import LGBOptimizer
from sklearn.datasets import make_regression, make_classification
from sklearn.metrics import *

X, y = make_classification(n_samples=1000)

opt = LGBOptimizer('/Users/yuanjie/Desktop/Projects/Python/aizoo/aizoo/tuner/search_space/lgb.yaml',
                   X, y, feval=roc_auc_score)

print(opt.search_space)
opt.optimize(3)
