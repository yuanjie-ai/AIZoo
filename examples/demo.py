#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : MeUtils.
# @File         : demo
# @Time         : 2021/9/4 下午3:26
# @Author       : yuanjie
# @WeChat       : 313303303
# @Software     : PyCharm
# @Description  :

import optuna


def objective(trial):
    trial.suggest_categorical('categorical', choices='abc')

    trial.suggest_int("int", low=0, high=1000, step=1, log=False)
    trial.suggest_int("log_int", low=1, high=1000, step=1, log=True)  # step=1 and log=True

    trial.suggest_float("uniform", low=0.1, high=1000, step=None, log=False)
    trial.suggest_float("loguniform", low=0.1, high=1000, step=None, log=True)  # step=None and log=True
    trial.suggest_float("discrete_uniform", low=0.1, high=1000, step=0.1, log=False)

    return 1


study = optuna.create_study(direction="maximize", study_name="LGBM Classifier")
study.optimize(objective, n_trials=1000)

study.trials_dataframe()  # .filter(regex="_uniform").plot()
