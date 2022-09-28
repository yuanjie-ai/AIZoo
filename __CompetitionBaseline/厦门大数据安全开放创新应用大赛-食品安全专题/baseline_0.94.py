#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : Python.
# @File         : baseline_0.94
# @Time         : 2022/9/28 上午10:31
# @Author       : yuanjie
# @WeChat       : meutils
# @Software     : PyCharm
# @Description  :


def submit(df, username, password):
    localpath = 'result.txt'
    df[['EVEN_ID', 'EVENT_TYPE', 'EVENT_NAME']].to_csv(localpath, '|', index=False)
    print(df.head())

    import paramiko

    with paramiko.Transport("sftp.ai.xm.gov.cn:57891") as sf:
        sf.connect(username=username, password=password)

        sftp = paramiko.SFTPClient.from_transport(sf)
        sftp.put(localpath, '/result/result.txt')


if __name__ == '__main__':
    import pandas as pd

    df = pd.read_excel('食品安全-算法分析题初赛A榜-综合信息数据.xls').fillna('')
    df['EVENT_TYPE'] = 0
    pattern = '食|餐|肉|饭|菜|面包|蛋糕'
    df.loc[(df.EVENT_NAME + df.CONTENT).str.contains(pattern), 'EVENT_TYPE'] = 1

    submit(df, '***', '***')
