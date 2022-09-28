#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : aizoo.
# @File         : demo
# @Time         : 2022/7/15 上午10:23
# @Author       : yuanjie
# @WeChat       : meutils
# @Software     : PyCharm
# @Description  :


from pinferencia import Server

# ME
from meutils.pipe import *


class MyModel:

    @ttl_cache(key=str)
    @staticmethod
    def predict(data):
        time.sleep(3)
        return data


model = MyModel()

service = Server()
service.register(model_name="mymodel", model=model, entrypoint="predict")
