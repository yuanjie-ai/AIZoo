![image](https://img.shields.io/pypi/v/aizoo.svg) ![image](https://img.shields.io/travis/Jie-Yuan/aizoo.svg) ![image](https://readthedocs.org/projects/aizoo/badge/?version=latest)

<h1 align = "center">🔥aizoo🔥</h1>

---
# Install
```
pip install aizoo -U
```

# [Docs](https://jie-yuan.github.io/aizoo)

# Usages
## OOF models
```python
from aizoo.tab.models import TabNetClassifier
from sklearn.datasets import make_regression, make_classification
from sklearn.metrics import r2_score, roc_auc_score

X, y = make_classification(n_samples=10000)

TabNetClassifier().run(X, y, feval=roc_auc_score)
```

## Hyperparameter optimization: [search_space][1]
```python
from aizoo.tuner.optimizers import LGBOptimizer, F1Optimizer
from sklearn.datasets import make_regression, make_classification

X, y = make_classification(n_samples=1000)
opt = LGBOptimizer('search_space.yaml',  X, y)
best_params = opt.optimize(100)
opt.plot()
```
![newplot](https://tva1.sinaimg.cn/large/008i3skNgy1guiih927a2j60rd0el75102.jpg)

---
# TODO

---
[1]: ./aizoo/tuner/search_space

