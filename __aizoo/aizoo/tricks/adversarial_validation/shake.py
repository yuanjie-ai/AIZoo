#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : DeepNN.
# @File         : shake
# @Time         : 2020/4/27 11:09 上午
# @Author       : yuanjie
# @Email        : yuanjie@xiaomi.com
# @Software     : PyCharm
# @Description  : 


# @jit(forceobj=True)
# def get_scores():
#     s = []
#     for n in tqdm_notebook(range(1300, 1800, 1)):
#         s.append(roc_auc_score([1]*n+[0]*(170610-n), [1]*1+[0]*(170610-1)))
#     return s