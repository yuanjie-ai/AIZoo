#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : aizoo.
# @File         : __init__.py
# @Time         : 2021/9/23 上午11:08
# @Author       : yuanjie
# @WeChat       : 313303303
# @Software     : PyCharm
# @Description  :


from itertools import combinations

# ME
from meutils.pipe import *


class FE(object):

    def __init__(self, cat_feats=None, num_feats=None, cat_funcs=None, num_funcs=None, agg_dict=None,
                 cat_feat_combination=None):
        """

        @param cat_feats: cat_feats、num_feats同不为None
        @param num_feats:
        @param cat_funcs: ['nunique', 'max', 'min'] 已经包含count计算
        @param num_funcs: ['min', 'mean', 'median', 'max', 'sum', 'std', 'var', 'sem', 'skew']
        @param agg_dict: 覆盖默认的funcs
            {
                'N0': ['mean']
            }

        @cat_feat_combination: 自定义特征组合，默认所有组合 [['C0'], ...]

        """
        self.cat_feats = cat_feats
        self.num_feats = num_feats

        self.cat_funcs = cat_funcs if cat_funcs is not None else ['nunique']
        self.num_funcs = num_funcs if num_funcs is not None else ['mean', 'std']

        self.agg_dict = agg_dict if agg_dict is not None else {}

        self.cat_feat_combination = cat_feat_combination

    def transform(self, df: pd.DataFrame, n_jobs=1, threshold=10000):
        """提前计算一下特征值数量，过滤

        @param df:
        @param n_jobs:
        @return:
        """

        if self.cat_feats is None:
            self.cat_feats = self.infer_cat_feats(df)
            logger.info(f"Infer cat feats: {self.cat_feats}")

            if self.num_feats is None:
                self.num_feats = df.columns.difference(self.cat_feats).tolist()
                logger.info(f"Infer num feats: {self.num_feats}")

        if self.cat_feat_combination is None:
            # self.cat_feat_combination = self.combination_all(self.cat_feats)
            self.cat_feat_combination_df = self.calculate_featvalue_num(df, self.cat_feats, n_jobs, threshold)
            self.cat_feat_combination = self.cat_feat_combination_df['keys'].tolist()

            ##############################################################
            # self.calculate_feature_number()
            ##############################################################

        bar = tqdm(self.cat_feat_combination, desc='agg🐢')

        if n_jobs == 1:
            for keys in bar:
                df_ = self.group_calculate(df, keys)
                df = df.merge(df_, 'left')  # df = df.merge(df_, on=keys)

        else:  # 多进程加速
            def func(keys):
                return self.group_calculate(df, keys)

            dfs = bar | xJobs(func, n_jobs) | xtqdm(desc=f'joining🐢')

            for df_ in dfs:
                df = df.merge(df_, 'left')  # todo: 并行设计

            #############################无加速效果#################################
            # def func(dfs):
            #     dff = df.copy()
            #     for df_ in dfs:
            #         dff = dff.merge(df_)
            #     return dff
            #
            # def dfs_join(l, func=func, n_jobs=n_jobs):
            #     batch_size = len(l) // n_jobs
            #
            #     if batch_size == 1:
            #         return func(l)
            #
            #     l = l | xgroup(batch_size) | xtqdm(desc=f'joining🐢bs={batch_size}') | xJobs(func, n_jobs)
            #
            #     return dfs_join(l)
            #
            # df = dfs_join(dfs)
            ##############################################################

        return df

    def group_calculate(self, df, keys):
        agg_dict = self._get_agg_dict(keys)
        agg_dict[keys[0]] = ['count']  # count只统计一次

        df_ = df.groupby(keys, dropna=False).agg(agg_dict)
        df_.columns = [f"{':'.join(keys)}.{func.upper()}({feat})" for feat, func in df_.columns]
        df_.reset_index(inplace=True)
        return df_

    def _get_agg_dict(self, keys):
        if self.num_funcs:
            num_agg_dict = {feat: self.num_funcs for feat in self.num_feats}
        else:
            num_agg_dict = {}  # num_funcs=[] 不聚合num特征

        if self.cat_funcs:
            cat_agg_dict = {feat: self.cat_funcs for feat in self.cat_feats if feat not in keys}
        else:
            cat_agg_dict = {}  # cat_funcs=[] 不聚合cat特征

        return {**cat_agg_dict, **num_agg_dict, **self.agg_dict}

    @staticmethod
    def combination_all(s):
        comb_list = []
        for i in range(1, len(s) + 1):
            comb_list += map(list, combinations(s, i))
        return comb_list

    def calculate_featvalue_num(self, df, cat_feats, n_jobs=6, threshold=10000):
        func = lambda keys: (df[keys].drop_duplicates().__len__(), len(keys), keys)

        _ = self.combination_all(cat_feats) | xtqdm(desc='计算特征值数🐢') | xJobs(func, n_jobs)
        df_ = (
            pd.DataFrame(_, columns=['num', 'keys_len', 'keys'])
                .sort_values('num', ascending=False, ignore_index=True)[lambda df: df['num'] <= threshold]
        )
        return df_

    # def calculate_featvalue_num(self, df, cat_feats, threshold=10000):
    #
    #     feat2num = df[cat_feats].nunique().to_dict()
    #
    #     df_ = pd.DataFrame(
    #         [
    #             (keys | xmap(lambda k: feat2num.get(k, 1)) | xreduce(lambda x, y: x * y), len(keys), keys)
    #             for keys in self.combination_all(cat_feats)
    #         ],
    #         columns=['num', 'keys_len', 'keys']
    #     ).sort_values('num', ascending=False, ignore_index=True)[lambda df: df['num'] <= threshold]
    #
    #     return df_

    @staticmethod
    def infer_cat_feats(df, threshold=64):
        threshold = min(len(df) // 100 + 8, threshold)
        return df.nunique()[lambda x: x < threshold].index.tolist()

    def calculate_feature_number(self):
        m = len(self.cat_feats)
        n = len(self.num_feats)

        cat_feat_num = ((2 ** m - 1) * m - sum(map(len, self.cat_feat_combination))) * len(self.cat_funcs) + \
                       len(self.cat_feat_combination)  # count特征数

        num_feat_num = (2 ** n - 1) * n * len(self.num_funcs)

        logger.info(f"CAT+NUM: {cat_feat_num + num_feat_num} = {cat_feat_num} + {num_feat_num}")


if __name__ == '__main__':
    df1 = pd.DataFrame({f"C{i}": np.random.randint(0, i + 3, size=100) for i in range(5)})
    df2 = pd.DataFrame(np.random.random((100, 5)), columns=(f"N{i}" for i in range(5)))

    df = pd.concat([df1, df2], 1)

    # ft = FE()
    # ft = FE(num_funcs=[])
    ft = FE(cat_funcs=[], num_funcs=[])

    df_ = ft.transform(df, n_jobs=2)

    print(df_.shape)

    print(df_.columns)
