[多输入多输出][1]
```python
import tensorflow.contrib.eager as tfe
tfe.enable_eager_execution()
np.set_printoptions(threshold=np.nan) # 输出所有元素
```
---
[1]: https://blog.csdn.net/u012938704/article/details/79904173