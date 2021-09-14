#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : aizoo.
# @File         : gensim_w2v
# @Time         : 2021/9/13 上午9:34
# @Author       : yuanjie
# @WeChat       : 313303303
# @Software     : PyCharm
# @Description  : 增量学习 https://mp.weixin.qq.com/s/VIHwIjliwSJecmG6RVYTiA


from meutils.pipe import *
from collections import Iterable
from gensim.models import word2vec


class Word2Vec(object):
    """
    架构：skip-gram（慢、对罕见字有利）vs CBOW（快）
    训练算法：分层softmax（对罕见字有利）vs 负采样（对常见词和低纬向量有利）
    负例采样准确率提高，速度会慢，不使用negative sampling的word2vec本身非常快，但是准确性并不高
    欠采样频繁词：可以提高结果的准确性和速度（适用范围1e-3到1e-5）
    文本（window）大小：skip-gram通常在10附近，CBOW通常在5附近
    size: n = sqrt(词汇量)/2

    - [skip-gram][1]: 1个学生(中心词) `汇报 =>` K个老师(周围词)
    - cbow: K个学生(周围词) `汇报 =>` 1个老师(中心词)

    """

    def fit(self, corpus, vector_size=300, window=5, epochs=10, min_count=1, sg=0, hs=0, negative=5):
        """
        gensim.models.word2vec.Word2Vec?
        w2v.wv.key_to_index
        w2v.build_vocab(self.corpus, update=True) # 增量训练
        w2v.init_sims(replace=True) # 对model进行锁定，并且据说是预载了相似度矩阵能够提高后面的查询速度，但是你的model从此以后就read only了

        @param corpus: tqdm(sentences, desc="SentencePreprocessing")
        @param vector_size:
        @param window:
        @param min_count:
        @param sg:
        @param hs:
        @param negative:
        @param epochs:
        @return:
        """
        if isinstance(corpus, Iterable):
            sentences = corpus
            corpus_file = None

        else:  # Path(corpus).is_file()
            corpus_file = corpus
            sentences = None

        self.w2v = word2vec.Word2Vec(
            sentences,
            corpus_file,
            vector_size=vector_size,
            window=window,
            epochs=epochs,
            min_count=min_count,
            negative=negative,
            sg=sg, hs=hs,
            workers=32
        )

        logger.info(f"Word2Vec: ({len(self.w2v.wv.key_to_index)}, {vector_size})")


if __name__ == '__main__':
    docs = [['Well', 'done!'],
            ['Good', 'work'],
            ['Great', 'effort'],
            ['nice', 'work'],
            ['Excellent!'],
            ['Weak'],
            ['Poor', 'effort!'],
            ['not', 'good'],
            ['poor', 'work'],
            ['Could', 'have', 'done', 'better.']]
    w2v = Word2Vec()
    w2v.fit(docs)
