#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : aizoo.
# @File         : data_agument
# @Time         : 2022/6/17 下午6:01
# @Author       : yuanjie
# @WeChat       : meutils
# @Software     : PyCharm
# @Description  : https://mp.weixin.qq.com/s/oZgvve4wad0alZngcYd_rg


"""
https://blog.csdn.net/weixin_44575152/article/details/118056405
https://blog.csdn.net/weixin_44575152/article/details/123779647
http://t.zoukankan.com/ghgxj-p-14219097.html
https://github.com/PaddlePaddle/ERNIE/tree/ernie-kit-open-v1.0/applications/tools/data/data_aug
"""

# import paddlenlp.dataaug
# https://github.com/PaddlePaddle/PaddleNLP/blob/develop/docs/dataaug.md

PUNCTUATIONS = ['。', ',', '!', '?', ';', ':']
PUNC_RATIO = 0.5

from meutils.pipe import *


def insert_punctuation_marks(sentence, punc_ratio=PUNC_RATIO, tokenizer=str.split):
    """该论文所提出的AEDA方法，主要是在原始文本中随机插入一些标点符号，属于增加噪声的一种，主要与EDA论文对标，突出“简单”二字。注意：该方法仅适用于文本分类任务。"""
    words = tokenizer(sentence)
    new_line = []
    q = random.randint(1, int(punc_ratio * len(words) + 1))
    qs = random.sample(range(0, len(words)), q)

    print(q)
    print(qs)

    for j, word in enumerate(words):
        if j in qs:
            new_line.append(PUNCTUATIONS[random.randint(0, len(PUNCTUATIONS) - 1)])
            new_line.append(word)
        else:
            new_line.append(word)
    new_line = ' '.join(new_line)
    return new_line

from trustai.demo import DEMO

from trustai.interpretation import IntGradInterpreter

if __name__ == '__main__':
    from LAC import LAC

    print(insert_punctuation_marks('人类语言是抽象的信息符号，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的含义。'))
