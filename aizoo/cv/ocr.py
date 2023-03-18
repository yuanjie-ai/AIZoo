#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : aizoo.
# @File         : ocr
# @Time         : 2022/9/29 下午4:05
# @Author       : yuanjie
# @WeChat       : meutils
# @Software     : PyCharm
# @Description  :


import os
import cv2
from paddleocr import PPStructure,draw_structure_result,save_structure_res, to_excel
from paddleocr.ppstructure.recovery.recovery_to_doc import sorted_layout_boxes, convert_info_docx

table_engine = PPStructure(show_log=True)

save_folder = './output'
img_path = 'tab.png'
img = cv2.imread(img_path)
result = table_engine(img) # return_ocr_result_in_table=True

save_structure_res(result, save_folder,os.path.basename(img_path).split('.')[0])


for line in result:
    line.pop('img')
    print(line)

from PIL import Image

font_path = './fonts/simfang.ttf' # PaddleOCR下提供字体包
image = Image.open(img_path).convert('RGB')
im_show = draw_structure_result(image, result,font_path=font_path)
im_show = Image.fromarray(im_show)
im_show.save('result.jpg')
