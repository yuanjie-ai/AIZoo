#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : DeepNN.
# @File         : config_utils
# @Time         : 2020/5/8 4:13 下午
# @Author       : yuanjie
# @Email        : yuanjie@xiaomi.com
# @Software     : PyCharm
# @Description  : 


from pyapollo import ApolloClient


class Config(object):
    """
    http://106.54.227.205/
    https://github.com/ctripcorp/apollo

    client = ApolloClient('400000008', 'c3', 'http://10.114.38.22:8080') # port不对
    """

    def __init__(self, ):
        pass

    def apollo_client(self, app_id='yuanjie', cluster='yuanjie', config_server_url="http://106.54.227.205:8080"):
        """
            client.get_value('yuanjie')
        :param app_id:
        :param cluster:
        :param config_server_url:
        :return:
        """
        client = ApolloClient(app_id, cluster, config_server_url)
        client.start()
        return client

if __name__ == '__main__':
    Config().apollo_client()