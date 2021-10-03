#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : aizoo.
# @File         : eda
# @Time         : 2021/9/30 下午6:46
# @Author       : yuanjie
# @WeChat       : 313303303
# @Software     : PyCharm
# @Description  : 探索性数据分析（Exploratory Data Analysis，EDA）


"""https://www.cnblogs.com/HuZihu/p/11146493.html
1. summary
2. train/test/目标值可视化分析:
    1）train与test分布是否一致【判别器、异常值、聚类】
    2）train特征与label的关系
3. 缺失值分析: missingno 与label的关系
4. 相关性分析
5. 异常值分析: 独立森林、箱型图、与label的关系
"""

from meutils.pipe import *


class EDA(object):

    def __init__(self):
        self.dfs = None

    def summary(self, df, transpose=False):
        from pandas_summary import DataFrameSummary

        self.dfs = DataFrameSummary(df)
        _ = self.dfs.summary()

        # del self.dfs.df

        return _ if transpose is False else _.T


if __name__ == '__main__':
    from sklearn.datasets import load_iris

    dfs = load_iris(return_X_y=True, as_frame=True)
    df = pd.concat(dfs, axis=1)

    eda = EDA()

    print(eda.summary(df))
    print(eda.summary(df, 1))
