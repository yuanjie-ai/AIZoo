#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : inn.
# @File         : baseline
# @Time         : 2020/5/19 5:33 下午
# @Author       : yuanjie
# @Email        : yuanjie@xiaomi.com
# @Software     : PyCharm
# @Description  : 

import tensorflow as tf
from inn.dateset import Dataset, load_heart
from inn.models import Baseline

df = load_heart(0)

df.head()

num_cols = ['age', 'sex']  # 选择部分列

ds = Dataset().from_cache(df[num_cols], df['target'], shuffle_seed=1)

model = Baseline(list(map(tf.feature_column.numeric_column, num_cols))).model

model.compile(loss='categorical_crossentropy')

tf.keras.utils.plot_model(model, show_shapes=True)

model.fit(ds, epochs=10, validation_data=ds)

# model.save('model.h5')
