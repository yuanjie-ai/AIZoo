#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : aizoo.
# @File         : common
# @Time         : 2022/4/1 下午5:22
# @Author       : yuanjie
# @WeChat       : 313303303
# @Software     : PyCharm
# @Description  :


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
