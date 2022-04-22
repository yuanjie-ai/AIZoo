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

<details><summary>Details</summary>

```py
def forward(self, input_ids, attention_mask, token_type_ids):
    out = self.extractor(input_ids,
                         attention_mask=attention_mask,
                         token_type_ids=token_type_ids,
                         output_hidden_states=True)

    first = out.hidden_states[1].transpose(1, 2)
    last = out.hidden_states[-1].transpose(1, 2)
    first_avg = torch.avg_pool1d(
        first, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
    last_avg = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(
        -1)  # [batch, 768]
    avg = torch.cat((first_avg.unsqueeze(1), last_avg.unsqueeze(1)),
                    dim=1)  # [batch, 2, 768]
    out = torch.avg_pool1d(avg.transpose(1, 2), kernel_size=2).squeeze(-1)
    x = self.fc(out)
    x = F.normalize(x, p=2, dim=-1)
    return x
 ```

</details>

![newplot](https://tva1.sinaimg.cn/large/008i3skNgy1guiih927a2j60rd0el75102.jpg)

---
# TODO

---
[1]: ./aizoo/tuner/search_space

