#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : inn.
# @File         : scheduler.py
# @Time         : 2020/5/18 2:28 下午
# @Author       : yuanjie
# @Email        : yuanjie@xiaomi.com
# @Software     : PyCharm
# @Description  : 


import os

os.system("pip install -U --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple loguru schedule")
import time
import schedule

from loguru import logger

logger.add('./scheduler.log',
           rotation="100 MB",
           enqueue=True,  # 异步
           encoding="utf-8",
           backtrace=True,
           diagnose=True,
           level="INFO")


def job():
    logger.info("I'm working...")
    from mi.db import Hive
    Hive()

    for _ in range(11):
        os.system("python /fds/seize.py")


#     _ = os.popen("ps aux | grep seize.py | awk '{print $2}'").read().split()
#     [os.system(f"kill -9 {i}") for i in _]


schedule.every().day.at("00:00").do(job)

while True:
    schedule.run_pending()
    time.sleep(1)
