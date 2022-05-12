#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : aizoo.
# @File         : tokenizer_utils
# @Time         : 2022/4/12 下午4:45
# @Author       : yuanjie
# @WeChat       : meutils
# @Software     : PyCharm
# @Description  :


from meutils.pipe import *

from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast


def tokenizer_encode(texts: Union[str, pd.Series, List[str], Tuple[str]],
                     tokenizer, max_length=128,
                     return_token_type_ids=True,
                     return_tensors='np',
                     **kwargs):
    """BertTokenizerFast.__doc__

    @param texts:
    @param tokenizer:
    @param max_length:
    @param return_tensors: np pt tf
    @return: dict_keys(['input_ids', 'token_type_ids', 'attention_mask'])

    """
    if isinstance(texts, str):
        texts = [texts]
    elif isinstance(texts, pd.Series):
        texts = texts.astype(str).tolist()

    return tokenizer.batch_encode_plus( # tokenizer()
        texts,
        max_length=max_length,
        truncation=True,  # add
        pad_to_max_length=True,
        return_attention_mask=True,
        return_token_type_ids=return_token_type_ids,
        return_tensors=return_tensors,
        **kwargs  # add_special_tokens
    )
