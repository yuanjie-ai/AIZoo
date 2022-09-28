#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : aizoo.
# @File         : __init__.py
# @Time         : 2021/9/13 上午9:31
# @Author       : yuanjie
# @WeChat       : 313303303
# @Software     : PyCharm
# @Description  :


from paddlespeech.cli.asr.infer import ASRExecutor

asr = ASRExecutor()
result = asr(audio_file="zh.wav")
