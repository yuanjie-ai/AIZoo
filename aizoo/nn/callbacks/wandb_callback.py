#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : Python.
# @File         : wandb_callback
# @Time         : 2020-03-13 12:48
# @Author       : yuanjie
# @Email        : yuanjie@xiaomi.com
# @Software     : PyCharm
# @Description  : https://github.com/wandb/client

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer

wandb_logger = WandbLogger(project="pl")
trainer = Trainer(logger=wandb_logger)

# 通用的可放外面

import wandb

# Step1: Initialize W&B run
wandb.init(project='project_name')

# 2. Save model inputs and hyperparameters
config = wandb.config





# token
# 789ae399af943555652e476ff1d0c0452ee86564
