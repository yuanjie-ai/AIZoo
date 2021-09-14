#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : aizoo.
# @File         : model_optimizers
# @Time         : 2021/9/14 下午12:39
# @Author       : yuanjie
# @WeChat       : 313303303
# @Software     : PyCharm
# @Description  :

import optuna
from lightgbm import LGBMClassifier


def LGBMClassifierTrial(params, trial: optuna.trial.Trial):
    opt_params = dict(
        num_leaves=trial.suggest_int("num_leaves", 2, 2 ** 8),
        learning_rate=trial.suggest_discrete_uniform('learning_rate', 0.001, 1, 0.001),
        n_estimators=trial.suggest_int("n_estimators", 2, 2 ** 10, log=True),

        min_child_samples=trial.suggest_int('min_child_samples', 2, 2 ** 8),
        min_child_weight=trial.suggest_loguniform('min_child_weight', 1e-8, 1),
        min_split_gain=trial.suggest_loguniform('min_split_gain', 1e-8, 1),

        subsample=trial.suggest_uniform('subsample', 0.4, 1),
        subsample_freq=trial.suggest_int("subsample_freq", 0, 2 ** 4),
        colsample_bytree=trial.suggest_uniform('colsample_bytree', 0.4, 1),
        reg_alpha=trial.suggest_loguniform('reg_alpha', 1e-8, 10),
        reg_lambda=trial.suggest_loguniform('reg_lambda', 1e-8, 10),
    )
    clf = LGBMClassifier(
        boosting_type='gbdt',
        num_leaves=31,
        max_depth=-1,
        learning_rate=0.1,
        n_estimators=100,
        subsample_for_bin=200000,
        objective=None,
        class_weight=None,
        min_split_gain=0.,
        min_child_weight=1e-3,
        min_child_samples=20,
        subsample=1.,
        subsample_freq=0,
        colsample_bytree=1.,
        reg_alpha=0.,
        reg_lambda=0.,
        random_state=None,
        n_jobs=-1,
        silent=True,
        importance_type='split'
    )
    clf.set_params(**{**opt_params, **params})
    return clf
