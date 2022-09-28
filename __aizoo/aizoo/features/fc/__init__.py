#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : inn.
# @File         : __init__.py
# @Time         : 2020/5/22 1:13 下午
# @Author       : yuanjie
# @Email        : yuanjie@xiaomi.com
# @Software     : PyCharm
# @Description  : 


from collections import namedtuple


class Column(object):
    pass


class NumericColumn(Column,
                    namedtuple('NumericColumn', ('name', 'shape', 'dtype', 'normalizer_fn'))):
    def __new__(cls, name, shape=(1,), normalizer_fn=None):
        _ = super().__new__(cls, name=name,
                            shape=shape,
                            dtype='float32',
                            normalizer_fn=normalizer_fn)
        return _


class CategoricalColumn(Column,
                        namedtuple('CategoricalColumn',
                                   ('name', 'shape', 'vocabulary_size', 'embedding_dim', 'embedding_name',
                                    'field', 'dtype'))):
    def __new__(cls, name='CategoricalColumn',
                vocabulary_size=1000,  # nunique
                embedding_dim=64,
                embedding_name=None,
                field='field'):

        if embedding_name is None:
            embedding_name = name

        if embedding_dim == 'auto' or embedding_dim is None:
            embedding_dim = int(6 * vocabulary_size ** 0.25)

        _ = super().__new__(cls, name=name,
                            vocabulary_size=vocabulary_size,
                            embedding_dim=embedding_dim,
                            embedding_name=embedding_name,
                            field=field,
                            dtype='int32',
                            shape=(1,))
        return _


class SequenceCategoricalColumn(Column,
                                namedtuple('SequenceCategoricalColumn',
                                           ('name', 'shape', 'maxlen', 'combiner',
                                            'vocabulary_size', 'embedding_dim', 'embedding_name',
                                            'length_name', 'weight_name',
                                            'weight_norm', 'field', 'dtype'))):
    def __new__(cls, name,
                maxlen,
                vocabulary_size,
                embedding_dim=None,
                embedding_name=None,
                combiner='mean',
                length_name="length_name",
                weight_name="weight_name",
                weight_norm=None,
                field='field'):

        if embedding_name is None:
            embedding_name = name

        if embedding_dim == 'auto' or embedding_dim is None:
            embedding_dim = int(6 * vocabulary_size ** 0.25)

        _ = super().__new__(cls, name=name,
                            maxlen=maxlen,
                            vocabulary_size=vocabulary_size,
                            embedding_dim=embedding_dim,
                            embedding_name=embedding_name,
                            combiner=combiner,
                            length_name=length_name,
                            weight_name=weight_name,
                            weight_norm=weight_norm,
                            field=field,
                            dtype='int32',
                            shape=(maxlen,))
        return _
