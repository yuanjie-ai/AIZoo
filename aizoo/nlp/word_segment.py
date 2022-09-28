#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : aizoo.
# @File         : word_segment
# @Time         : 2022/7/19 下午5:04
# @Author       : yuanjie
# @WeChat       : meutils
# @Software     : PyCharm
# @Description  :


from aizoo.nlp.paddlenlp_utils import *


class WordSegment(object):

    def __init__(self, mode='lac'):
        """

        @param mode:
        """
        self.mode = mode
        self.taskflow = Taskflow('word_segmentation', model=None, mode=None)

