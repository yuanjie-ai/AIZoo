#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : Python.
# @File         : sentence_embedding
# @Time         : 2022/11/1 上午10:34
# @Author       : yuanjie
# @WeChat       : meutils
# @Software     : PyCharm
# @Description  :


import torch
from roformer import RoFormerForCausalLM, RoFormerConfig
from transformers import BertTokenizer, BertTokenizerFast

# ME
from meutils.pipe import *


class SentenceEmbedding(object):

    def __init__(self, pretrained_model=get_resolve_path("models/junnyu/roformer_chinese_sim_char_ft_small", __file__),
                 device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model, self.tokenizer = self.get_model_and_tokenizer(str(pretrained_model))

    @torch.no_grad()
    def encode(self, texts):
        encode_input = self.tokenizer(texts, return_tensors="pt", padding=True, max_length=512, truncation=True)
        V = self.model(**encode_input.to(self.device)).pooler_output
        # V = V.to(self.device)
        V /= (V ** 2).sum(axis=1, keepdims=True) ** 0.5
        return V.cpu().numpy()

    def get_model_and_tokenizer(self, pretrained_model):
        if pretrained_model.__contains__('roformer'):
            tokenizer = BertTokenizerFast.from_pretrained(pretrained_model)
            config = RoFormerConfig.from_pretrained(
                pretrained_model,
                is_decoder=True,
                eos_token_id=tokenizer.sep_token_id,
                pooler_activation="linear"
            )
            model = RoFormerForCausalLM.from_pretrained(pretrained_model, config=config)
            model.to(self.device)
            model.eval()
            return model, tokenizer

        elif pretrained_model.__contains__('text2vec'):  # pretrained_model="shibing624/text2vec-base-chinese"

            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(pretrained_model)
            return model, None


if __name__ == '__main__':
    se = SentenceEmbedding()
    print(se.encode(['中国人']))
