#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : DeepTricks.
# @File         : __init__.py
# @Time         : 2019-10-21 13:00
# @Author       : yuanjie
# @Email        : yuanjie@xiaomi.com
# @Software     : PyCharm
# @Description  : 

import pandas as pd
from sklearn.model_selection import train_test_split
from meutils.aizoo.dateset.Dataset import Dataset


def load_iris(batch_size=128):
    from sklearn import datasets
    ds = Dataset(batch_size)
    X, y = datasets.load_iris(True)
    return ds.from_cache(X, y)


def load_heart(return_ds=True):
    df = pd.read_csv('https://storage.googleapis.com/applied-dl/heart.csv')
    if return_ds:
        train, test = train_test_split(df, test_size=0.2)
        ds_train = Dataset().from_cache(train.drop('target', 1), train['target'])
        ds_test = Dataset().from_cache(test.drop('target', 1), test['target'])
        return ds_train, ds_test
    else:
        return df


if __name__ == '__main__':
    print(load_heart())