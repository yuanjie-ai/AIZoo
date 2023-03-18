#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = 'info_value'
__author__ = 'JieYuan'
__mtime__ = '19-3-12'
"""

from meutils.pipe import *


class InformationValue(object):
    """
        小于0.02预测能力无
        大于0.30预测能力强
    """

    def __init__(self, df: pd.DataFrame, label: str):
        """
        :param df:
        :param label: target name
        """
        assert label in df.columns

        self.label = label
        self.df = df.assign(_label=1 - df[label])
        self.feats = [col for col in df.columns if col != label]

        self.y1 = self.df[label].values.sum()
        self.y0 = self.df['_label'].values.sum()

    @property
    def iv(self, ):
        df = pd.DataFrame([(feat, self._iv(feat)) for feat in self.feats], columns=['iv', 'feats'])

        return df.sort_values('feats', ascending=False, ignore_index=True)

    def _iv(self, feat):
        gr = self.df.groupby(feat)
        gr1, gr0 = gr[self.label].sum().values + 1e-8, gr['_label'].sum().values + 1e-8
        good, bad = gr1 / self.y1, gr0 / self.y0
        woe = np.log(good / bad)
        iv = (good - bad) * woe
        return iv.sum()
