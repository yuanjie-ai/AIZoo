#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : aizoo.
# @File         : paddlenlp_utils
# @Time         : 2022/7/7 下午3:20
# @Author       : yuanjie
# @WeChat       : meutils
# @Software     : PyCharm
# @Description  :


from paddlenlp.taskflow.taskflow import Taskflow as _Taskflow, TASKS

# ME
from meutils.pipe import *

Taskflow = lru_cache()(_Taskflow)
"""
word_segmentation: fast/base/accurate
"""

def taskflow4batch(data, batch_size=64, taskflow=Taskflow('ner'), cache=None):
    """

    @param data: 批量数据
    @param batch_size:
    @param taskflow:
    @param cache: 默认硬盘缓存
    @return:
    """
    if isinstance(cache, str):
        taskflow = disk_cache(location=cache)(taskflow.__call__)

    return data | xgroup(batch_size) | xtqdm | xmap(taskflow) | xchain | xlist


def text_correction_view(item):
    """
    {'source': '人生就是如此，经过磨练才能让自己更加拙壮，才能使自己更加乐观。',
     'target': '人生就是如此，经过磨练才能让自己更加茁壮，才能使自己更加乐观。',
     'errors': [{'position': 18, 'correction': {'拙': '茁'}}]}
    """
    s = item['source']
    l = list(s)
    for error in item['errors']:  # 多个插入会有问题
        l.insert(error['position'] + 1, str(list(error['correction'].values())))

    return ''.join(l)


if __name__ == '__main__':
    taskflow4batch(['人生就是如此，经过磨练才能让自己更加拙壮，才能使自己更加乐观。'])
