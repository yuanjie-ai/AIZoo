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

from aizoo.tuner.base import optuna, Tuner
from aizoo.tab.models import LGBMOOF
from aizoo.utils.check_utils import check_classification


class F1Optimizer(Tuner):

    def __init__(self, search_space, y, y_pred, **kwargs):
        super().__init__(search_space, **kwargs)
        self.y = y
        self.y_pred = y_pred

    def objective(self, trial: optuna.trial.Trial):
        params = self.trial_choice(trial)

        y_pred_ = np.where(np.array(self.y_pred) > params['threshold'], 1, 0)
        return f1_score(self.y, y_pred_)


class LGBOptimizer(Tuner):

    def __init__(self, search_space, X, y, feval, fit_params=None, oof_fit_params=None, **kwargs):
        """

        @param search_space:
        @param X:
        @param y:
        @param feval:
        @param fit_params: 原生fit参数
        @param oof_fit_params: dict(sample_weight=None, X_test=None, feval=None, cv=5, split_seed=777, target_index=None)
        @param kwargs:
        """
        super().__init__(search_space, **kwargs)

        self.X = X
        self.y = y
        self.feval = feval
        self.fit_params = fit_params
        self.oof_fit_params = oof_fit_params if oof_fit_params is not None else {}

    def objective(self, trial: optuna.trial.Trial):
        params = self.trial_choice(trial)

        task = 'Classifier' if check_classification(self.y) else 'Regressor'

        _ = (
            LGBMOOF(params=params, fit_params=self.fit_params, task=task)
                .fit(self.X, self.y, feval=self.feval, **self.oof_fit_params)
        )

        if _ is None:
            raise ValueError("Target is None⚠️")
        return _


if __name__ == '__main__':
    y = [1, 1, 0, 0]
    y_pred = [0.1, 0.2, 0.3, 0.4]

    # opt = F1Optimizer({'threshold': 0.1}, y, y_pred)
    opt = F1Optimizer("./search_space/f1.yaml", y, y_pred)

    opt.optimize(
        100,
        direction='minimize',
        study_name='test',
        storage="sqlite:////Users/yuanjie/Desktop/Projects/Python/aizoo/aizoo/tuner/test.db",
        load_if_exists=True  # cli --skip-if-exists
    )
    # optuna-dashboard sqlite:////Users/yuanjie/Desktop/Projects/Python/aizoo/aizoo/tuner/test.db

    print(opt.trials_dataframe)
