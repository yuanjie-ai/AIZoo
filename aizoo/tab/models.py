#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : aizoo.
# @File         : oof
# @Time         : 2021/9/14 下午3:42
# @Author       : yuanjie
# @WeChat       : 313303303
# @Software     : PyCharm
# @Description  : todo: 增加nn模型


from aizoo.base import OOF


class LGBMClassifier(OOF):

    def fit_predict(self, X_train, y_train, X_valid, y_valid, X_test, **kwargs):
        import lightgbm as lgb

        clf = lgb.LGBMClassifier()
        if self.params is not None:
            clf.set_params(**self.params)

        eval_set = [(X_train, y_train), (X_valid, y_valid)]
        self.clf = clf.fit(
            X_train, y_train,
            eval_set=eval_set,
            eval_metric=None,
            eval_names=('Train', 'Valid'),
            verbose=100,
            early_stopping_rounds=100,  # fit_params
            **kwargs  # TODO: set_params
        )
        if hasattr(clf, 'predict_proba'):
            return clf.predict_proba(X_valid), clf.predict_proba(X_test)
        else:
            return clf.predict(X_valid), clf.predict(X_test)


class LGBMRegressor(OOF):

    def fit_predict(self, X_train, y_train, X_valid, y_valid, X_test, **kwargs):
        import lightgbm as lgb

        clf = lgb.LGBMRegressor()
        if self.params is not None:
            clf.set_params(**self.params)

        eval_set = [(X_train, y_train), (X_valid, y_valid)]
        self.clf = clf.fit(
            X_train, y_train,
            eval_set=eval_set,
            eval_metric=None,
            eval_names=('Train', 'Valid'),
            verbose=100,
            early_stopping_rounds=100,  # fit_params
            **kwargs  # TODO: set_params
        )

        if hasattr(clf, 'predict_proba'):
            return clf.predict_proba(X_valid), clf.predict_proba(X_test)
        else:
            return clf.predict(X_valid), clf.predict(X_test)


class XGBClassifier(OOF):

    def fit_predict(self, X_train, y_train, X_valid, y_valid, X_test, **kwargs):
        import xgboost as xgb

        clf = xgb.XGBClassifier()
        if self.params is not None:
            clf.set_params(**self.params)

        eval_set = [(X_train, y_train), (X_valid, y_valid)]
        self.clf = clf.fit(
            X_train, y_train,
            eval_set=eval_set,
            eval_metric=None,
            verbose=100,
            early_stopping_rounds=100
        )

        if hasattr(clf, 'predict_proba'):
            return clf.predict_proba(X_valid), clf.predict_proba(X_test)
        else:
            return clf.predict(X_valid), clf.predict(X_test)


class XGBRegressor(OOF):

    def fit_predict(self, X_train, y_train, X_valid, y_valid, X_test, **kwargs):
        import xgboost as xgb

        clf = xgb.XGBRegressor()
        if self.params is not None:
            clf.set_params(**self.params)

        eval_set = [(X_train, y_train), (X_valid, y_valid)]
        self.clf = clf.fit(
            X_train, y_train,
            eval_set=eval_set,
            eval_metric=None,
            verbose=100,
            early_stopping_rounds=100
        )

        if hasattr(clf, 'predict_proba'):
            return clf.predict_proba(X_valid), clf.predict_proba(X_test)
        else:
            return clf.predict(X_valid), clf.predict(X_test)


class CatBoostClassifier(OOF):

    def fit_predict(self, X_train, y_train, X_valid, y_valid, X_test, **kwargs):
        import catboost as cab

        clf = cab.CatBoostClassifier(thread_count=30)  # TODO: embedding_features
        if self.params is not None:
            clf.set_params(**self.params)

        # eval_set = [(X_train, y_train), (X_valid, y_valid)]
        self.clf = clf.fit(X_train, y_train,
                           eval_set=(X_valid, y_valid),  # CatBoostError: Multiple eval sets are not supported on GPU
                           # Only one of parameters ['verbose', 'logging_level', 'verbose_eval', 'silent'] should be set
                           verbose=100,
                           early_stopping_rounds=100,
                           use_best_model=True,
                           plot=True,
                           **kwargs
                           )
        # evals_result = self.clf.evals_result()

        if hasattr(clf, 'predict_proba'):
            return clf.predict_proba(X_valid), clf.predict_proba(X_test)
        else:
            return clf.predict(X_valid), clf.predict(X_test)


class CatBoostRegressor(OOF):

    def fit_predict(self, X_train, y_train, X_valid, y_valid, X_test, **kwargs):
        import catboost as cab

        clf = cab.CatBoostRegressor(thread_count=30)  # TODO: embedding_features
        if self.params is not None:
            clf.set_params(**self.params)

        # eval_set = [(X_train, y_train), (X_valid, y_valid)]
        self.clf = clf.fit(X_train, y_train,
                           eval_set=(X_valid, y_valid),  # CatBoostError: Multiple eval sets are not supported on GPU
                           # Only one of parameters ['verbose', 'logging_level', 'verbose_eval', 'silent'] should be set
                           verbose=100,
                           early_stopping_rounds=100,
                           use_best_model=True,
                           plot=True,
                           **kwargs
                           )
        # evals_result = self.clf.evals_result()

        if hasattr(clf, 'predict_proba'):
            return clf.predict_proba(X_valid), clf.predict_proba(X_test)
        else:
            return clf.predict(X_valid), clf.predict(X_test)


class TabNetClassifier(OOF):

    def fit_predict(self, X_train, y_train, X_valid, y_valid, X_test, **kwargs):
        from pytorch_tabnet import tab_model

        clf = tab_model.TabNetClassifier()
        if self.params is not None:
            clf.set_params(**self.params)

        eval_set = [(X_train, y_train), (X_valid, y_valid)]
        self.clf = clf.fit(
            X_train, y_train,
            eval_set=eval_set,
            eval_metric=None,
            eval_name=('Train', 'Valid'),
            max_epochs=100,
            patience=5,
        )
        if hasattr(clf, 'predict_proba'):
            return clf.predict_proba(X_valid), clf.predict_proba(X_test)
        else:
            return clf.predict(X_valid), clf.predict(X_test)


class TabNetRegressor(OOF):

    def fit_predict(self, X_train, y_train, X_valid, y_valid, X_test, **kwargs):
        from pytorch_tabnet import tab_model

        clf = tab_model.TabNetRegressor()
        if self.params is not None:
            clf.set_params(**self.params)

        y_train = y_train.reshape(-1, 1)
        y_valid = y_valid.reshape(-1, 1)  # nn输入的不同
        eval_set = [(X_train, y_train), (X_valid, y_valid)]
        self.clf = clf.fit(
            X_train, y_train,
            eval_set=eval_set,
            eval_metric=None,
            eval_name=('Train', 'Valid'),
            max_epochs=100,
            patience=5,
        )
        if hasattr(clf, 'predict_proba'):
            return clf.predict_proba(X_valid).reshape(-1), clf.predict_proba(X_test).reshape(-1)
        else:
            return clf.predict(X_valid).reshape(-1), clf.predict(X_test).reshape(-1)  # (1000, 1)


if __name__ == '__main__':
    from sklearn.metrics import r2_score, roc_auc_score
    from sklearn.datasets import make_regression, make_classification

    for Model in [TabNetClassifier, CatBoostClassifier, LGBMClassifier, XGBClassifier]:
        X, y = make_classification(n_samples=1000)
        oof = Model()
        oof.run(X, y, feval=roc_auc_score, cv=5)
        break

    for Model in [TabNetRegressor, CatBoostRegressor, LGBMRegressor, XGBRegressor]:
        X, y = make_regression(n_samples=1000)
        oof = Model()
        oof.run(X, y, feval=r2_score, cv=5)
        break
