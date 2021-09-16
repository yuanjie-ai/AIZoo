#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : aizoo.
# @File         : optimizers
# @Time         : 2021/9/16 上午11:00
# @Author       : yuanjie
# @WeChat       : 313303303
# @Software     : PyCharm
# @Description  :


from meutils.pipe import *
from sklearn.metrics import *

from aizoo.tuner.base import Tuner
from aizoo.tab.models import LGBMClassifier


class F1Optimizer(Tuner):

    def __init__(self, search_space, y, y_pred, **kwargs):
        super().__init__(search_space, **kwargs)
        self.y = y
        self.y_pred = y_pred

    def trainer(self, params, **kwargs):
        y_pred_ = np.where(np.array(self.y_pred) > params['threshold'], 1, 0)

        return f1_score(self.y, y_pred_)

class LGBOptimizer(Tuner):

    def __init__(self, search_space, X, y, feval=roc_auc_score, **kwargs):
        super().__init__(search_space, **kwargs)
        self.X = X
        self.y = y
        self.feval = feval

    def trainer(self, params, **kwargs):

        _ = LGBMClassifier(params).run(self.X, self.y, feval=self.feval, **self.kwargs)

        return _


if __name__ == '__main__':
    y = [1, 1, 0, 0]
    y_pred = [0.1, 0.2, 0.3, 0.4]

    # opt = F1Optimizer({'threshold': 0.1}, y, y_pred)
    opt = F1Optimizer("./search_space/f1.yaml", y, y_pred)

    opt.optimize()

    print(opt.trials_dataframe)
