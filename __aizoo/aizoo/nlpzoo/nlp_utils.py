#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : MeUtils.
# @File         : nlp_utils
# @Time         : 2020/11/19 10:45 上午
# @Author       : yuanjie
# @Email        : yuanjie@xiaomi.com
# @Software     : PyCharm
# @Description  : 

from meutils.pipe import *
from meutils.np_utils import normalize


def sent2vec(sent, w2v, tokenizer=str.split):
    return normalize(np.row_stack([w2v[w] for w in tokenizer(sent)]).mean(0))
