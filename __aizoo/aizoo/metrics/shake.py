#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : DeepTricks.
# @File         : shake
# @Time         : 2019-09-14 23:27
# @Author       : yuanjie
# @Email        : yuanjie@xiaomi.com
# @Software     : PyCharm
# @Description  : 

import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import numba
import numpy as np


class Shake(object):

    def __init__(self, oof_preds, y_true, feval=roc_auc_score):
        self.oof_preds = oof_preds
        self.y_true = y_true
        self.feval = feval

    def plot_difference(self, n_splits=100, test_size=0.7):
        self.cal(n_splits, test_size)
        self.public_private = np.array(self.metric_public) - np.array(self.metric_private)

        plt.figure(figsize=(10, 6))
        plt.hist(self.public_private, bins=50)
        plt.title('(Public - Private) scores')
        plt.xlabel(f'{self.feval} score difference')
        plt.show()

    @numba.jit()
    def cal(self, n_splits=100, test_size=0.7):
        self.metric_public = []
        self.metric_private = []
        for rs in tqdm(range(n_splits)):
            _ = train_test_split(self.oof_preds, self.y_true, test_size=test_size, random_state=rs)
            y_preds_public, y_preds_private, y_public, y_private = _
            self.metric_public.append(self.feval(y_public, y_preds_public))
            self.metric_private.append(self.feval(y_private, y_preds_private))
