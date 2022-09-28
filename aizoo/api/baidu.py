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

from aip import AipNlp, AipOcr

keys = """
15696121
md9ONR0cj5pvF9oxYlg9MIMg
iYGkDYuW4XDGXjMtXPZclUplLhiBNBgQ
"""
# 25308860 GmAPqlyBDNLaoAqO2mrFhjS2 vXIoxFUdx2jiuRnGLvZSDMEczEaZsc1K
client = AipNlp(*keys.split())
ocr_client = AipOcr(*keys.split())

# ocr_client.taiwanExitentrypermit

import requests
from pprint import pprint
url = 'https://tva1.sinaimg.cn/large/e6c9d24egy1h36evn3pcoj20af06tq3c.jpg'
# url = 'https://img0.baidu.com/it/u=3690939738,1965122121&fm=253&fmt=auto&app=138&f=JPEG?w=499&h=320'
# url = 'https://img0.baidu.com/it/u=1750169569,767502721&fm=253&fmt=auto&app=120&f=JPEG?w=404&h=241'
image = requests.get(url).content

_ = ocr_client.HKMacauExitentrypermit(image=image)
pprint(_)

_ = ocr_client.taiwanExitentrypermit(image=image)
pprint(_)

