#!/usr/bin/env bash
# @Project      : MeUtils
# @Time         : 2021/4/23 1:41 下午
# @Author       : yuanjie
# @Email        : yuanjie@xiaomi.com
# @Software     : PyCharm
# @Description  : ${DESCRIPTION}

MODEL_NAME=baseline_model
source=/Users/yuanjie/Desktop/Projects/Python/MeUtils/meutils/aizoo/examples/baseline_model/1
target=/models/$MODEL_NAME/1
docker run -p 8800:8800 \
--mount type=bind,source=$source,target=$target \
-e MODEL_NAME=$MODEL_NAME \
-t tensorflow/serving

