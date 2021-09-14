#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : aizoo.
# @File         : baidu
# @Time         : 2021/9/13 下午4:36
# @Author       : yuanjie
# @WeChat       : 313303303
# @Software     : PyCharm
# @Description  : https://ai.baidu.com/ai-doc/IMAGESEARCH/Pk3bczxau

"""
client.commentTag
client.depParser
client.dnnlm
client.ecnet
client.emotion
client.keyword
client.lexer
client.lexerCustom
client.newsSummary
client.sentimentClassify
client.simnet
client.topic
client.wordEmbedding
client.wordSimEmbedding
"""

from aip import AipNlp

keys = """
15696121
md9ONR0cj5pvF9oxYlg9MIMg
iYGkDYuW4XDGXjMtXPZclUplLhiBNBgQ
"""
client = AipNlp(*keys.split())
