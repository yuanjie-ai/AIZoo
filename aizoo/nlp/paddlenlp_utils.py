#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : aizoo.
# @File         : paddlenlp_utils
# @Time         : 2022/7/7 下午3:20
# @Author       : yuanjie
# @WeChat       : meutils
# @Software     : PyCharm
# @Description  :


from paddlenlp.taskflow.taskflow import Taskflow as _Taskflow, TASKS
from paddle.io import DataLoader, BatchSampler, DistributedBatchSampler
from paddlenlp.data import DataCollatorWithPadding
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import AutoModelForSequenceClassification, AutoTokenizer, LinearDecayWithWarmup
# ME
from meutils.pipe import *


def text_correction_view(item):
    """
    {'source': '人生就是如此，经过磨练才能让自己更加拙壮，才能使自己更加乐观。',
     'target': '人生就是如此，经过磨练才能让自己更加茁壮，才能使自己更加乐观。',
     'errors': [{'position': 18, 'correction': {'拙': '茁'}}]}
    """
    s = item['source']
    l = list(s)
    for error in item['errors']:  # 多个插入会有问题
        l.insert(error['position'] + 1, str(list(error['correction'].values())))

    return ''.join(l)


def dataloader_from_dataframe(df, batch_size, max_seq_length, tokenizer, shuffle=True):
    def _read(df):
        yield from df.to_dict('r')

    # line = next(_read(df))  # 判断label

    ds = (
        load_dataset(_read, lazy=False, df=df)
            .map(lambda example: {
            **tokenizer(**example, max_seq_len=max_seq_length),
            **{'labels': np.array([example['label']], dtype='int64')}  # TODO: 有label
        }) # batch['input_ids'], batch['token_type_ids'], batch['labels']
    )
    batch_sampler = BatchSampler(ds, batch_size=batch_size, shuffle=shuffle)
    collate_fn = DataCollatorWithPadding(tokenizer)
    return DataLoader(dataset=ds, batch_sampler=batch_sampler, collate_fn=collate_fn)


if __name__ == '__main__':
    # taskflow4batch(['人生就是如此，经过磨练才能让自己更加拙壮，才能使自己更加乐观。'])
    pass
