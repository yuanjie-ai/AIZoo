#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : aizoo.
# @File         : agg_feat
# @Time         : 2021/10/1 下午1:56
# @Author       : yuanjie
# @WeChat       : 313303303
# @Software     : PyCharm
# @Description  :

from itertools import combinations

# ME
from meutils.pipe import *


class FE(object):

    def __init__(self, cat_feats=None, num_feats=None, cat_funcs=None, num_funcs=None, agg_dict=None):
        """

        @param cat_feats:
        @param num_feats:
        @param cat_funcs: ['nunique', 'max', 'min']
        @param num_funcs: ['min', 'mean', 'median', 'max', 'sum', 'std', 'var', 'sem', 'skew']
        @param trans_dict: 覆盖默认的funcs
            {
                'cat1': ['nunique']
            }
        """
        self.cat_feats = cat_feats
        self.num_feats = num_feats

        self.cat_funcs = cat_funcs if cat_funcs is not None else ['nunique']
        self.num_funcs = num_funcs if num_funcs is not None else ['mean', 'std']

        self.agg_dict = agg_dict if agg_dict is not None else {}

    def transform(self, df: pd.DataFrame):
        if self.cat_feats is None:
            self.cat_feats = self.infer_cat_feats(df)
            logger.warning(f"Infer cat feats: {self.cat_feats}")

            if self.num_feats is None:
                self.num_feats = df.columns.difference(self.cat_feats).tolist()
                logger.warning(f"Infer cat feats: {self.num_feats}")

        for keys in tqdm(self.combination_all(self.cat_feats), desc='🐢'):
            agg_dict = self._get_agg_dict(keys)
            agg_dict[keys[0]] = ['count']  # count只统计一次

            df_ = df.groupby(keys).agg(agg_dict)
            df_.columns = [f"{':'.join(keys)}.{func.upper()}({feat})" for feat, func in df_.columns]
            df_.reset_index(inplace=True)
            df = df.merge(df_, on=keys)

        return df

    def _get_agg_dict(self, keys):
        num_agg_dict = {feat: self.num_funcs for feat in self.num_feats}
        cat_agg_dict = {feat: self.cat_funcs for feat in self.cat_feats if feat not in keys}
        return {**cat_agg_dict, **num_agg_dict, **self.agg_dict}

    @staticmethod
    def combination_all(s):
        comb_list = []
        for i in range(1, len(s) + 1):
            comb_list += map(list, combinations(s, i))
        return comb_list

    @staticmethod
    def infer_cat_feats(df, threshold=64):
        threshold = min(len(df) // 100 + 8, threshold)
        return df.nunique()[lambda x: x < threshold].index.tolist()


if __name__ == '__main__':
    df1 = pd.DataFrame({f"C{i}": np.random.randint(0, i + 3, size=100) for i in range(5)})
    df2 = pd.DataFrame(np.random.random((100, 5)), columns=(f"N{i}" for i in range(5)))

    df = pd.concat([df1, df2], 1)

    ft = FE()
    df_ = ft.transform(df)

    print(df_)
