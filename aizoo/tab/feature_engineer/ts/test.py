#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : Python.
# @File         : test
# @Time         : 2022/10/19 下午4:41
# @Author       : yuanjie
# @WeChat       : meutils
# @Software     : PyCharm
# @Description  :


import datetime

# 判断 2018年4月30号 是不是节假日
from chinese_calendar import is_holiday, is_workday
april_last = datetime.date(2018, 4, 30)
assert is_workday(april_last) is False
assert is_holiday(april_last) is True

# 或者在判断的同时，获取节日名
import chinese_calendar as calendar  # 也可以这样 import
on_holiday, holiday_name = calendar.get_holiday_detail(april_last)
assert on_holiday is True
assert holiday_name == calendar.Holiday.labour_day.value

# 还能判断法定节假日是不是调休
import chinese_calendar
assert chinese_calendar.is_in_lieu(datetime.date(2006, 2, 1)) is False
assert chinese_calendar.is_in_lieu(datetime.date(2006, 2, 2)) is True

print(holiday_name)
