#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : MeUtils.
# @File         : grpc
# @Time         : 2021/4/23 7:35 下午
# @Author       : yuanjie
# @WeChat       : 313303303
# @Software     : PyCharm
# @Description  : 

import grpc
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc

from meutils.pipe import *

# mi
from cloud_ml_sdk.utils.load_balance_dns import DnsLoadBanlance, request_serving
from cloud_ml_sdk.client import CloudMlClient


class GrpcClient(object):

    def __init__(self, ak=None, sk=None, endpoint=None, model='yuanjie', version=1, signature_name='serving_default'):
        if LOCAL:
            with open('/Users/yuanjie/.config/xiaomi/config') as f:
                cfg = json.load(f)['xiaomi_cloudml']['yuanjie']
                ak = cfg['xiaomi_access_key_id']
                sk = cfg['xiaomi_secret_access_key']
                endpoint = cfg['xiaomi_cloudml_endpoint']

        self.client = CloudMlClient(access_key=ak, secret_key=sk, endpoint=endpoint)
        self.model_info = self.client.describe_model_service(model_name=model, model_version=version)  # todo

        self.dns_ip = self.model_info["dns_ip"]
        self.grpc_address = self.model_info["grpc_address"]
        self.service_port = self.model_info["service_port"]

        self.dnsload = DnsLoadBanlance(
            dns_ip=self.dns_ip,
            grpc_address=self.grpc_address,
            port=self.service_port,
            func=grpc.insecure_channel
        )
        # 构造数据
        self.request = predict_pb2.PredictRequest()
        self.request.model_spec.name = model
        self.request.model_spec.signature_name = signature_name

    def predict(self, data=tf.make_tensor_proto([[1.0] * 5])):
        self.request.inputs["dense_input"].CopyFrom(data)  # shape=[1,5]

        return request_serving(self.dnsload, prediction_service_pb2_grpc.PredictionServiceStub, "Predict", self.request)


if __name__ == '__main__':
    print(GrpcClient().predict())
