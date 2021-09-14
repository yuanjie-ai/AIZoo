![image](https://img.shields.io/pypi/v/aizoo.svg) ![image](https://img.shields.io/travis/Jie-Yuan/aizoo.svg) ![image](https://readthedocs.org/projects/aizoo/badge/?version=latest)

<h1 align = "center">🐎 aizoo 🐎</h1>

---
# Install
```
pip install aizoo -U
```

# [Docs](https://jie-yuan.github.io/aizoo)

# Usages
```python
from aizoo.tab.models import TabNetClassifier
from sklearn.datasets import make_regression, make_classification
from sklearn.metrics import r2_score, roc_auc_score

X, y = make_classification(n_samples=10000)

TabNetClassifier().run(X, y, feval=roc_auc_score)
```
---
* TODO

封装 lazypredict
