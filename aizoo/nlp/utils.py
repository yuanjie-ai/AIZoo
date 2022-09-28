#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : aizoo.
# @File         : utils
# @Time         : 2022/7/14 下午3:32
# @Author       : yuanjie
# @WeChat       : meutils
# @Software     : PyCharm
# @Description  :

from meutils.pipe import *

from LAC import LAC
def get_nouns(texts):
    lac = LAC()
    dic = {'NOUN':[], 'PER': [], 'LOC': [], 'ORG': [], 'TIME': []}
    for words, flags in lac.run(texts):
        for i, flag in enumerate(flags):
            if 'n' in flag:
                dic['NOUN'].append(words[i])
            elif flag in dic:
                dic[flag].append(words[i])

    return dic
