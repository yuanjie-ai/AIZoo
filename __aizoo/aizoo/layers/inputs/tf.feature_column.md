# [tf.feature_column](https://mp.weixin.qq.com/s/5jfOahNKnUjTre0O2655IA)

## `numeric_column`
- ``
```python
from tensorflow.feature_column import *
numeric_column(key="dense_feature", shape=(1,), normalizer_fn=None)
bucketized_column
```

## `categorical_column_with_* + indicator_column/embedding_column`
```python
from tensorflow.feature_column import *

categorical_column_with_vocabulary_list # one-hot，最常用，后面可接embedding层
categorical_column_with_vocabulary_file # 从file获得类别list, 按行存储
categorical_column_with_identity  # 按索引分箱
categorical_column_with_hash_bucket # hash转换：categorical_column_with_identity升级版
weighted_categorical_column


feature_column = categorical_column_with_vocabulary_list('sex', ['0', '1'])
indicator_column(feature_column)
embedding_column(feature_column, 2)
```

## `weighted_categorical_column`
```python
color_data = {'color': [['R'], ['G'], ['B'], ['A']],
                  'weight': [[1.0], [2.0], [4.0], [8.0]]}  # 4行样本
color_column = tf.feature_column.categorical_column_with_vocabulary_list(
        'color', ['R', 'G', 'B'], dtype=tf.string, default_value=-1)
color_weight_categorical_column = weighted_categorical_column(color_column, 'weight')
feature_column = tf.feature_column.indicator_column(color_weight_categorical_column)

show_column(feature_column, color_data)

```
## `crossed_column`
```python
import tensorflow as tf
featrues = {
        'price': [['A', 'A'], ['B', 'D'], ['C', 'A']],
        'color': [['R', 'R'], ['G', 'G'], ['B', 'B']]
    }

price = tf.feature_column.categorical_column_with_vocabulary_list('price',
                                                               ['A', 'B', 'C', 'D'])
color = tf.feature_column.categorical_column_with_vocabulary_list('color',
                                                               ['R', 'G', 'B'])
p_x_c = tf.feature_column.crossed_column([price, color], 16)

p_x_c_identy = tf.feature_column.indicator_column(p_x_c)

show_column(featrues, p_x_c_identy)
```

## `sequence_categorical_column_with_*`
> https://github.com/wangjunbo2000/wangjb/blob/master/tensorflow/python/feature_column/sequence_feature_column_test.py
```python
color_data = {'colors': [['R', 'Y'], ['G', 'Y'], ['B', 'B'], ['A', 'B']],
 'weight': [[1.0], [2.0], [4.0], [8.0]]}

fc = tf.feature_column.sequence_categorical_column_with_vocabulary_list(
    key='colors', vocabulary_list=('R', 'G', 'B', 'Y'),
    num_oov_buckets=2)

e = tf.feature_column.embedding_column(fc, dimension=10)
show_column(color_data, e)


tf.feature_column.sequence_numeric_column???


```




## `embedding_column/shared_embeddings`
> 共享向量空间
```python
def model_fn(features, ...):
    watched_video_id = categorical_column_with_vocabulary_file(
        'watched_video_id', video_vocabulary_file, video_vocabulary_size)
    impression_video_id = categorical_column_with_vocabulary_file(
        'impression_video_id', video_vocabulary_file, video_vocabulary_size)
    columns = shared_embedding_columns(
        [watched_video_id, impression_video_id], dimension=10)
    dense_tensor = input_layer(features, columns)
    # Form DNN layers, calculate loss, and return EstimatorSpec.
  ...
```
