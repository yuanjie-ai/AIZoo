# #!/usr/bin/env python
# # -*- coding: utf-8 -*-
# # @Project      : inn.
# # @File         : Emebedding
# # @Time         : 2020/5/18 4:12 下午
# # @Author       : yuanjie
# # @Email        : yuanjie@xiaomi.com
# # @Software     : PyCharm
# # @Description  :
#
#
import tensorflow as tf

tf.sparse.to_dense

#
# from tensorflow.keras.layers import *
#
#
# def get_embedding(vocabulary_size, embedding_dim, maxlen):
#     """
#       Input shape:
#     2D tensor with shape: `(batch_size, input_length)`.
#
#   Output shape:
#     3D tensor with shape: `(batch_size, input_length, output_dim)`.
#     """
#
#     Embedding(input_dim=vocabulary_size,
#               output_dim=embedding_dim,
#               embeddings_initializer,
#               embeddings_regularizer,
#               name='embedding',
#               mask_zero
#               input_length=maxlen)
