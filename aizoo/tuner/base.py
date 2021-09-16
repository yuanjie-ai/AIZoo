#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : aizoo.
# @File         : base
# @Time         : 2021/9/16 上午10:12
# @Author       : yuanjie
# @WeChat       : 313303303
# @Software     : PyCharm
# @Description  :

import optuna
from optuna.samplers import TPESampler

from meutils.pipe import *

optuna.logging.set_verbosity(optuna.logging.ERROR)


class Tuner(object):

    def __init__(self, search_space, **kwargs):
        """

        @param search_space: Union[str, dict]
        @param kwargs:
        """
        self.search_space = self.search_space_from_yaml(search_space)
        self.kwargs = kwargs

    @abstractmethod
    def trainer(self, params, **kwargs):
        raise NotImplementedError("overwrite objective!!!")

    def _objective(self, trial: optuna.trial.Trial):
        params = self.choice(self.search_space, trial)
        return self.trainer(params, **self.kwargs)

    def optimize(self, trials=3,
                 direction='maximize',
                 sampler=TPESampler(seed=222),
                 study_name='optimizer',
                 pruner=None,
                 gc_after_trial=False):

        self.study = optuna.create_study(study_name=study_name, direction=direction, sampler=sampler, pruner=pruner)
        self.study.optimize(
            self._objective,
            n_trials=trials,
            gc_after_trial=gc_after_trial,
            show_progress_bar=True
        )

        return self.study.best_params

    def plot_history(self):
        # optuna.visualization.plot_slice(self.study)
        _ = optuna.visualization.plot_optimization_history(self.study)

        # optuna.visualization.plot_param_importances(self.study)
        # optuna.visualization.plot_edf(self.study)
        # optuna.visualization.plot_parallel_coordinate(self.study)
        return _

    @property
    def trials_dataframe(self):
        df_trials = self.study.trials_dataframe().sort_values('value', ascending=False)

        return df_trials

    def top_params(self, topK=5, save_file=None):
        params_dict = (
            self.trials_dataframe.iloc[:topK, :]
                .filter(like='params_')
                .rename(columns=lambda col: col.split('params_')[1])
                .to_dict('records')
        )
        if save_file is not None:
            with open(save_file, 'w') as f:
                print(params_dict, file=f)

        return params_dict

    @staticmethod
    def search_space_from_yaml(search_space: Union[str, dict]):
        if isinstance(search_space, dict):
            return search_space
        elif isinstance(search_space, str) and Path(search_space).is_file():
            with open(search_space) as f:
                return yaml.safe_load(f)
        else:
            raise Exception('load params error')

    @staticmethod
    def choice(search_space: dict, trial: optuna.trial.Trial):
        params = {}
        for k, v in search_space.items():
            if isinstance(v, dict):
                v = v.copy()
                suggest_type = v.pop('type') if 'type' in v else v.pop('suggest_type')

                choice_func = trial.__getattribute__(f"suggest_{suggest_type}")
                params[k] = choice_func(k, **v)
            # elif isinstance(v, (int, float, bool, List[int], List[float])):
            elif isinstance(v, (str, int, float, bool, list)) or v is None:
                # choices = v if isinstance(v, (List[int], List[float], List[bool])) else [v]
                choices = v if isinstance(v, list) else [v]
                params[k] = trial.suggest_categorical(k, choices)  # [None]

            else:
                raise TypeError(f'suggest_type error: {k}_{v}')
        return params
