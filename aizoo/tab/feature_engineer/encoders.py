#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : aizoo.
# @File         : encoders
# @Time         : 2021/9/23 上午11:11
# @Author       : yuanjie
# @WeChat       : 313303303
# @Software     : PyCharm
# @Description  :


from collections import OrderedDict

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CountEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, dropna=False, normalize=False):
        """

        :param dropna: 缺失值是否计数，默认计数
        :param normalize: 频数还是频率，默认频数
        """
        self.dropna = dropna
        self.normalize = normalize
        self.mapper = None

    def fit(self, y):
        self.mapper = (pd.Series(y).value_counts(self.normalize, dropna=self.dropna)
                       .to_dict(OrderedDict))
        return self

    def transform(self, y):
        """不在训练集的补0，不经常出现补0"""
        return pd.Series(y).map(self.mapper).fillna(0)


class CountRankEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, topn=None):
        """

        :param topn: 仅保留topn个类别
        """
        self.topn = topn
        self.mapper = None

    def fit(self, y):
        ce = pd.Series(y).value_counts(True, dropna=False)  # 计数编码
        if self.topn:
            ce = ce[:self.topn]
            print(f"Coverage: {ce.sum() * 100:.2f}%")

        self.mapper = ce.rank(method='first').to_dict(OrderedDict)  # rank 合理？
        return self

    def transform(self, y):
        """不在训练集的补0，不经常出现补0"""
        return pd.Series(y).map(self.mapper).fillna(0)

class RankEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, method='average', na_option='keep'):
        """
        method : {'average', 'min', 'max', 'first', 'dense'}, default 'average'
            How to rank the group of records that have the same value
            (i.e. ties):

            * average: average rank of the group
            * min: lowest rank in the group
            * max: highest rank in the group
            * first: ranks assigned in order they appear in the array
            * dense: like 'min', but rank always increases by 1 between groups
        numeric_only : bool, optional
            For DataFrame objects, rank only numeric columns if set to True.
        na_option : {'keep', 'top', 'bottom'}, default 'keep'
            How to rank NaN values:

            * keep: assign NaN rank to NaN values
            * top: assign smallest rank to NaN values if ascending
            * bottom: assign highest rank to NaN values if ascending
        ascending : bool, default True
            Whether or not the elements should be ranked in ascending order.
        """
        self.method = method
        self.na_option = na_option

    def transform(self, y):
        """不在训练集的补0，不经常出现补0"""
        return pd.Series(y).rank(method=self.method, na_option=self.na_option)  # .fillna(0)

if __name__ == '__main__':
    import numpy as np

    s = ['a', 'a', 'b', 'b', 'c'] + [np.nan] * 6
    re = CountRankEncoder()

    print(re.fit_transform(s))
    print(re.mapper)


    s = ['a', 'a', 'a', 'b', 'b'] + [np.nan] * 10
    ce = CountEncoder()
    print(ce.fit_transform(s))
    print(ce.mapper)
