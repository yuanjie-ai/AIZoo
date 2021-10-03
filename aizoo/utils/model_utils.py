#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : aizoo.
# @File         : model_utils
# @Time         : 2021/9/30 下午7:57
# @Author       : yuanjie
# @WeChat       : 313303303
# @Software     : PyCharm
# @Description  :

from meutils.pipe import *


def get_imp(estimator, X=None, importance_type=None):
    """
    todo: 采样加速，可多次采样取平均，近似全量数据的shap值
    """
    if importance_type is not None:
        importance_type = importance_type.lower()
    if importance_type in (None, 'tree', 'gini', 'split', 'gain'):
        imp = estimator.feature_importances_

    elif importance_type in 'shap':
        import shap
        explainer = shap.Explainer(estimator)
        shap_values = explainer(X).values
        imp = np.abs(shap_values).mean(0)

        if shap_values.ndim == 3:
            imp = imp.sum(1)

        # if isinstance(coefs, list):
        #     coefs = list(map(lambda x: np.abs(x).mean(0), coefs))
        #     coefs = np.sum(coefs, axis=0)
        # else:
        #     coefs = np.abs(coefs).mean(0)

    elif importance_type in 'permutaion':
        imp = ...  # todo

    else:
        raise ValueError('No Importances was specified select one of (shap, tree/gini/split, permutaion)')

    return imp
