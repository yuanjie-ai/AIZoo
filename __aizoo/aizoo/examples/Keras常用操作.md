
```python
tf.keras.layers.Reshape((2, 3))(np.array([[1,2,3,4,5,6]]))

<tf.Tensor: shape=(1, 2, 3), dtype=int64, numpy=
array([[[1, 2, 3],
        [4, 5, 6]]])>
```
- Lambda
- Masking: 用于对值为指定值的位置进行掩蔽的操作，以忽略对应的timestep。
- Reshape
- Permute: 调换维度
- RepeatVector

- 变长序列
https://www.pianshen.com/article/8866331982/