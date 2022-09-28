#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : MeUtils.
# @File         : voice2vec
# @Time         : 2020/12/30 2:22 下午
# @Author       : yuanjie
# @Email        : yuanjie@xiaomi.com
# @Software     : PyCharm
# @Description  : https://github.com/philipperemy/deep-speaker


import librosa
from panns_inference import AudioTagging, SoundEventDetection, labels


class Voice2vec(object):

    def __init__(self, device='cuda'):
        self.at = AudioTagging(checkpoint_path=None, device=device)
        # self.sed = SoundEventDetection(checkpoint_path=None, device='cuda')

    def get_embedding(self, audio_path):
        audio = self.get_audio(audio_path)
        (clipwise_output, embedding) = self.at.inference(audio)
        return embedding

    @staticmethod
    def get_audio(audio_path):
        audio, _ = librosa.core.load(audio_path, sr=32000, mono=True)
        return audio[None, :]  # (batch_size, segment_samples)
