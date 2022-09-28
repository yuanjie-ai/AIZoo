#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : MeUtils.
# @File         : client
# @Time         : 2021/4/23 6:17 下午
# @Author       : yuanjie
# @WeChat       : 313303303
# @Software     : PyCharm
# @Description  : 

from meutils.pipe import *


class Client(object):

    def __init__(self, model='keras-input', url=None, version=1, mode='http'):
        self.model = model
        self.version = version
        self.mode = mode

        url = url if url else 'http://ms-33103-keras-input-2-0426203455.c5-cloudml.xiaomi.srv'

        self.url = f"{url}/v1/models/{model}/versions/{version}"
        self.model_version_status = requests.get(self.url).json()
        self.metadata_url = f"{self.url}/metadata"
        self.metadata = requests.get(self.metadata_url).json()
        self.info = self.metadata['metadata']['signature_def']['signature_def']['serving_default']
        self.inputs_info = self.info['inputs']
        self.outputs_info = self.info['outputs']

    def predict(self, data=None):
        if data is None:
            data = {
                "inputs": [[1, 2, 3, 4, 5], [1, 1, 1, 1, 1]],
                # "instances": [[1, 2, 3, 4, 5], [1, 1, 1, 1, 1]], # values

            }

        if self.mode == 'http':
            return requests.post(f'{self.url}:predict', json=data).json()


if __name__ == '__main__':
    c = Client()

    print(c.model_version_status)
    print(c.metadata)

    # outputs
    c.predict(data={'inputs': [[1, 2, 3, 4, 5], [1, 1, 1, 1, 1]]})
    c.predict(data={'inputs': {'myinput': [[1, 2, 3, 4, 5], [1, 1, 1, 1, 1]]}})

    # predictions
    c.predict(data={'instances': [[1, 2, 3, 4, 5], [1, 1, 1, 1, 1]]})
    c.predict(data={'instances': [{'myinput': [1, 1, 1, 1, 1]}, {'myinput': [1, 1, 1, 1, 1]}]})
