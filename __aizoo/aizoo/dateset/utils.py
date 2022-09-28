#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : DeepTricks.
# @File         : utils
# @Time         : 2020-02-19 11:46
# @Author       : yuanjie
# @Email        : yuanjie@xiaomi.com
# @Software     : PyCharm
# @Description  :

import pandas as pd


def libsvm2df(file, max_index=16, batch_size=128):
    dic = dict(zip(range(max_index), [0] * max_index))
    fn = lambda r: {**dic, **eval("{0:" + r.replace(' ', ',') + "}")}
    df_iter = pd.read_csv(file, names='r', iterator=True)
    yield pd.DataFrame(df_iter.get_chunk(batch_size)['r'].map(fn).values.tolist())


