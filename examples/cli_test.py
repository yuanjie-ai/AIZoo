#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : MeUtils.
# @File         : demo
# @Time         : 2021/9/4 下午3:26
# @Author       : yuanjie
# @WeChat       : 313303303
# @Software     : PyCharm
# @Description  :

from meutils.pipe import *


class TrainConfig(BaseConfig):
    a: int = "555"


@cli.command('x')
def train(p='config.yml'):
    conf = TrainConfig.parse_yaml(p)
    print(conf.dict())
    return conf.dict()





if __name__ == '__main__':
    cli()
