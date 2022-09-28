#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : mi.
# @File         : video
# @Time         : 2020/8/31 4:26 下午
# @Author       : yuanjie
# @Email        : yuanjie@xiaomi.com
# @Software     : PyCharm
# @Description  : 

from meutils.pipe import *
from moviepy.editor import *


# TODO: 抽帧去重
def video2picture(video='/fds/1_Work/2_DownVideo/videos/互联网_yidian_V_07d0oZik.mp4', top_duration=180):
    p = Path(video)
    pic_dir = ''.join(p.name[::-1].split('.')[-1:])[::-1]
    (p.parent / pic_dir).mkdir(exist_ok=True)

    with VideoFileClip(video) as clip:
        duration = int(clip.duration)
        for i in tqdm(range(min(duration, top_duration))):
            clip_ = clip.subclip(i, i + 1)
            clip_.save_frame(p.parent / pic_dir / f'{i}.png')


def video2audio(path_pair, verbose=False, subclip=None, ffmpeg_params=["-f", "mp3"]):
    """
        clip = VideoFileClip('蛋清和蛋黄是这么分离的.720p').subclip(3, 7)

    :param paths: (video_path, audio_path)
    :param subclip:
    :param ffmpeg_params:
    :return:
    """
    video_path, audio_path = path_pair

    with VideoFileClip(video_path) as clip:
        duration = int(clip.duration)
        if subclip:
            s, e = subclip[0], duration if subclip is None or duration < subclip[1] else subclip[1]
            clip = clip.subclip(s, e)

        clip.audio.write_audiofile(
            audio_path, fps=None, nbytes=2, buffersize=2000,
            codec=None, bitrate=None, ffmpeg_params=ffmpeg_params,
            write_logfile=False, verbose=verbose, logger='bar' if verbose else None
        )
