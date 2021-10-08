#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : aizoo.
# @File         : oof
# @Time         : 2021/9/14 下午3:42
# @Author       : yuanjie
# @WeChat       : 313303303
# @Software     : PyCharm
# @Description  : todo: 增加nn模型

# ME
from aizoo.base import OOF


class LGBMOOF(OOF):

    def _fit(self, X_train, y_train, w_train, X_valid, y_valid, w_valid, X_test, **kwargs):
        import lightgbm as lgb

        estimator = lgb.__getattribute__(f'LGBM{self.task}')()  # 实例
        estimator.set_params(**self.params)

        fit_params = dict(
            eval_set=[(X_train, y_train), (X_valid, y_valid)],
            eval_metric=None,
            eval_names=('Train🔥', 'Valid'),
            verbose=100,
            early_stopping_rounds=100,
            sample_weight=w_train,
            eval_sample_weight=[w_train, w_valid]  # 与eval_set一致
        )
        return estimator, fit_params


class XGBOOF(OOF):
    def _fit(self, X_train, y_train, w_train, X_valid, y_valid, w_valid, X_test, **kwargs):
        import xgboost as xgb

        estimator = xgb.__getattribute__(f'XGB{self.task}')()  # 实例
        estimator.set_params(**self.params)

        fit_params = dict(
            eval_set=[(X_train, y_train), (X_valid, y_valid)],
            eval_metric=None,
            verbose=100,
            early_stopping_rounds=100,
            sample_weight=w_train,
            sample_weight_eval_set=[w_train, w_valid]  # 与eval_set一致
        )
        return estimator, fit_params


class CatBoostOOF(OOF):

    def _fit(self, X_train, y_train, w_train, X_valid, y_valid, w_valid, X_test, **kwargs):
        import catboost as cab
        estimator = cab.__getattribute__(f'CatBoost{self.task}')()  # TODO: embedding_features
        estimator.set_params(**self.params)

        fit_params = dict(
            eval_set=(X_valid, y_valid),  # CatBoostError: Multiple eval sets are not supported on GPU
            verbose=100,
            early_stopping_rounds=100,
            sample_weight=w_train,

            use_best_model=True,
            plot=True,
        )

        return estimator, fit_params


class TabNetOOF(OOF):

    def _fit(self, X_train, y_train, w_train, X_valid, y_valid, w_valid, X_test, **kwargs):
        if self.task == 'Regressor':  # tabnet 回归输入的不同
            y_train = y_train.reshape(-1, 1)
            y_valid = y_valid.reshape(-1, 1)

        from pytorch_tabnet import tab_model

        estimator = tab_model.__getattribute__(f'TabNet{self.task}')()  # TODO: embedding_features
        estimator.set_params(**self.params)

        fit_params = dict(
            eval_set=[(X_train, y_train), (X_valid, y_valid)],
            eval_name=('Train🔥', 'Valid'),
            eval_metric=None,
            max_epochs=100,
            patience=5
        )

        return estimator, fit_params


if __name__ == '__main__':
    import shap
    from sklearn.metrics import r2_score, roc_auc_score
    from sklearn.datasets import make_regression, make_classification

    for Model in [LGBMOOF]:
        X, y = make_classification(n_samples=1000)
        oof = Model(importance_type='split')
        oof.fit(X, y, feval=roc_auc_score, cv=5)
        print(oof.predict(X))

        break

    # for Model in [LGBMOOF, TabNetRegressor, CatBoostRegressor, , XGBRegressor]:
    #     X, y = make_regression(n_samples=1000)
    #     oof = Model(weight_func=lambda w: 1 / (w + 1))
    #     oof.run(X, y, feval=r2_score, cv=5)
    #     break
