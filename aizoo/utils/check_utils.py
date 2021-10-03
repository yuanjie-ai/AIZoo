#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : aizoo.
# @File         : check_utils
# @Time         : 2021/9/27 下午9:18
# @Author       : yuanjie
# @WeChat       : 313303303
# @Software     : PyCharm
# @Description  :


from meutils.pipe import *


def check_classification(y, threshold=128):
    """

    @param y:
    @param threshold: 目标值多于 threshold 个就认为是回归问题
    @return:
    """
    return len(set(y)) < threshold


def check_pandas_input(self, arg):
    try:
        return arg.values
    except AttributeError:
        raise ValueError(
            "input needs to be a numpy array or pandas data frame."
        )


