#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : aizoo.
# @File         : demo
# @Time         : 2022/7/15 上午11:28
# @Author       : yuanjie
# @WeChat       : meutils
# @Software     : PyCharm
# @Description  :


this_is_an_example = 49
this_is_another_example = 49


def get_var_name(variable, local_dic):
    for k, v in local_dic.items():
        # if id(variable) == id(v):
        #     return k
        if variable == v:
            return k


vname = get_var_name(this_is_an_example, locals())
print(vname)

print(locals())
