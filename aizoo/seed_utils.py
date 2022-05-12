#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : DeepNN.
# @File         : seed_utils
# @Time         : 2020/4/21 1:15 下午
# @Author       : yuanjie
# @Email        : yuanjie@xiaomi.com
# @Software     : PyCharm
# @Description  :

import os
import random
import numpy as np
import torch


def set_seed(seed=2022):
    os.environ['PYTHONHASHSEED'] = '0'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    random.seed(seed)
    np.random.seed(seed)


def seed4torch(seed=2022):
    set_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = True  # Benchmark 模式会提升计算速度，但是由于计算中有随机性，每次网络前馈结果略有差异。
        torch.backends.cudnn.deterministic = True  # 避免这种结果波动
    else:
        torch.manual_seed(seed)


def seed4tf(seed=666):
    set_seed(seed)

    tf.compat.v1.reset_default_graph()
    tf.random.set_seed(seed)
