#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : DeepNN.
# @File         : __init__.py
# @Time         : 2020/4/21 1:39 下午
# @Author       : yuanjie
# @Email        : yuanjie@xiaomi.com
# @Software     : PyCharm
# @Description  : 

import os

if os.environ.get('NN') == 'torch':  # torch/tensorflow
    from .FM_ import FM
else:
    from .FM import FM
