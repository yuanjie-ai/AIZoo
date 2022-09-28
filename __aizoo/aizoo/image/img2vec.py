#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : MeUtils.
# @File         : image2vec
# @Time         : 2020/12/30 2:47 下午
# @Author       : yuanjie
# @Email        : yuanjie@xiaomi.com
# @Software     : PyCharm
# @Description  : https://github.com/milvus-io/bootcamp/blob/db5dfd63dc052ca15bdf28e7c6b7a8d510dd15f6/solutions/pic_search/webserver/src/preprocessor/vggnet.py#L19

from meutils.pipe import *
from meutils.np_utils import normalize
from meutils.aizoo.image import utils
from tensorflow.keras.preprocessing import image
from tensorflow.keras import applications


class Img2vec:
    def __init__(self, input_shape=(256, 256, 3), weights='imagenet', pooling='max', model='vgg16'):
        self.input_shape = input_shape

        self.model = getattr(applications, model.upper())(
            weights=weights,
            input_shape=input_shape,
            pooling=pooling,
            include_top=False
        )
        self.preprocess_input = getattr(applications, model).preprocess_input

        # TEST
        self.model.predict(np.zeros((1, *self.input_shape)), batch_size=1)

    def encoder(self, p):
        """
        todo: 增加多进程读取图片，支持图片url
        :param p:
        :return:
        """
        img = self._process_one_image(p)  # (1, 80, 256, 3) # (n, 80, 256, 3)
        img = self.preprocess_input(img)

        vecs = self.model.predict(img)
        return normalize(vecs)

    def _process_one_image(self, img_path):
        img = utils.load_img(img_path, target_size=self.input_shape[:2])
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)  # 1张照片
        return img


if __name__ == '__main__':
    url = 'https://img2.baidu.com/it/u=135896716,3704103640&fm=26&fmt=auto&gp=0.jpg'
    img2vec = Img2vec()
    print(img2vec.encoder(url))