# 自定义层

如果自定义模型层没有需要被训练的参数，一般推荐使用Lamda层实现。
Lamda层由于没有需要被训练的参数，只需要定义正向传播逻辑即可，使用比Layer基类子类化更加简单。
Lamda层的正向逻辑可以使用Python的lambda函数来表达，也可以用def关键字定义函数来表达。
```python
import tensorflow as tf
from tensorflow.keras import layers,models,regularizers

mypower = layers.Lambda(lambda x: tf.math.pow(x,2))

mypower(tf.range(5))
```

如果自定义模型层有需要被训练的参数，则可以通过对Layer基类子类化实现。
Layer的子类化一般需要重新实现初始化方法，Build方法和Call方法。下面是一个简化的线性层的范例，类似Dense.



# 然后2倍扩大或缩小，实验几次，一般就能得到一个相对较好的值。另外，embedding的初始化也是非常重要的，需要精细调参。
k * int(word_size**0.25) # k <= 16


# /Users/yuanjie/Desktop/Projects/Python/DeepCTR/tests/layers

