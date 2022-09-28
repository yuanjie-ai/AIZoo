#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : aizoo.
# @File         : TfidfClassifier
# @Time         : 2022/8/3 下午3:51
# @Author       : yuanjie
# @WeChat       : meutils
# @Software     : PyCharm
# @Description  :
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

# ME
from meutils.pipe import *


class TfidfClassifier(object):

    def __init__(self, estimator='sgd', vectorizer=TfidfVectorizer()):
        self._vectorizer = vectorizer
        self._estimator = make_pipeline(self._vectorizer, self._get_estimator(estimator))

    def fit(self, X, y=None, **fit_params):
        return self._estimator.fit(X, y, **fit_params)

    def predict(self, X):
        return self._estimator.predict_proba(X)

    @staticmethod
    def _get_estimator(estimator):
        if isinstance(estimator, str):
            if estimator == 'sgd':
                return SGDClassifier()
            elif estimator == 'lr':
                return LogisticRegression()
            elif estimator == 'nb':
                return MultinomialNB()
            else:
                logger.error('Unknown estimator')
        return estimator
