#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : DeepNN.
# @File         : FM
# @Time         : 2020/4/21 11:28 上午
# @Author       : yuanjie
# @Email        : yuanjie@xiaomi.com
# @Software     : PyCharm
# @Description  : 


#############################################################################################################
# Created by qqgeogor
# https://www.kaggle.com/qqgeogor
#############################################################################################################

import numpy as np
from sklearn.base import BaseEstimator
from keras.layers import Input, Embedding, Dense, Flatten, merge, Activation
from keras.models import Model
from keras.regularizers import l2 as l2_reg
import itertools


def build_model(max_features, K=8, solver='adam', l2=0.0, l2_fm=0.0):
    inputs = []
    flatten_layers = []
    columns = range(len(max_features))
    for c in columns:
        inputs_c = Input(shape=(1,), dtype='int32', name='input_%s' % c)
        num_c = max_features[c]

        embed_c = Embedding(
            num_c,
            K,
            input_length=1,
            name='embed_%s' % c,
            W_regularizer=l2_reg(l2_fm)
        )(inputs_c)

        flatten_c = Flatten()(embed_c)

        inputs.append(inputs_c)
        flatten_layers.append(flatten_c)

    fm_layers = []
    for emb1, emb2 in itertools.combinations(flatten_layers, 2):
        dot_layer = merge([emb1, emb2], mode='dot', dot_axes=1)
        fm_layers.append(dot_layer)

    for c in columns:
        num_c = max_features[c]
        embed_c = Embedding(
            num_c,
            1,
            input_length=1,
            name='linear_%s' % c,
            W_regularizer=l2_reg(l2)
        )(inputs[c])

        flatten_c = Flatten()(embed_c)

        fm_layers.append(flatten_c)

    flatten = merge(fm_layers, mode='sum')
    outputs = Activation('sigmoid', name='outputs')(flatten)

    model = Model(input=inputs, output=outputs)

    model.compile(
        optimizer=solver,
        loss='binary_crossentropy'
    )

    return model


class KerasFM(BaseEstimator):
    def __init__(self, max_features=[], K=8, solver='adam', l2=0.0, l2_fm=0.0):
        self.model = build_model(max_features, K, solver, l2=l2, l2_fm=l2_fm)

    def fit(self, X, y, batch_size=128, nb_epoch=10, shuffle=True, verbose=1, validation_data=None):
        self.model.fit(X, y, batch_size=batch_size, nb_epoch=nb_epoch, shuffle=shuffle, verbose=verbose,
                       validation_data=None)

    def fit_generator(self, X, y, batch_size=128, nb_epoch=10, shuffle=True, verbose=1, validation_data=None,
                      callbacks=None):
        tr_gen = batch_generator(X, y, batch_size=batch_size, shuffle=shuffle)
        if validation_data:
            X_test, y_test = validation_data
            te_gen = batch_generator(X_test, y_test, batch_size=batch_size, shuffle=False)
            nb_val_samples = X_test[-1].shape[0]
        else:
            te_gen = None
            nb_val_samples = None

        self.model.fit_generator(
            tr_gen,
            samples_per_epoch=X[-1].shape[0],
            nb_epoch=nb_epoch,
            verbose=verbose,
            callbacks=callbacks,
            validation_data=te_gen,
            nb_val_samples=nb_val_samples,
            max_q_size=10
        )

    def predict(self, X, batch_size=128):
        y_preds = predict_batch(self.model, X, batch_size=batch_size)
        return y_preds