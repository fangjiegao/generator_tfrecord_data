# coding=utf-8
import json
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import cv2


def process_points(im_path, points, output_path):
    # 获取坐标信息,
    points = np.array([points], dtype=np.int32)
    # 读取图片名
    img = cv2.imread(im_path)
    # # 绘制mask
    # zeros = np.zeros((img.shape), dtype=np.uint8)
    # 原本thickness = -1表示内部填充,这里不知道为什么会报错,只好不填充了
    cv2.polylines(img, points, isClosed=True, thickness=5, color=(144, 238, 144))
    cv2.imshow("img", img)
    cv2.waitKey(0)


def process_points_fill(im_path, points, output_path):
    # 获取坐标信息,
    points = np.array([points], dtype=np.int32)
    # 读取图片名
    img = cv2.imread(im_path)
    # # 绘制mask
    zeros = np.zeros((img.shape), dtype=np.uint8)
    # 原本thickness = -1表示内部填充
    mask = cv2.fillPoly(zeros, points, color=(1, 128, 255))
    # mask = zeros
    mask_img = mask + img
    cv2.imshow("img", mask_img)
    cv2.waitKey(0)


# 这是我用labelme画的不规则点坐标
points = [[14.222222222222229, 1076.111111111111], [133.1111111111111, 849.4444444444445],
          [240.8888888888889, 776.1111111111111], [252.0, 737.2222222222222],
          [253.1111111111111, 619.4444444444445], [298.66666666666663, 569.4444444444445],
          [298.66666666666663, 530.5555555555555], [270.8888888888889, 483.88888888888886],
          [247.55555555555554, 355.0], [242.0, 221.66666666666666], [282.0, 188.33333333333331],
          [332.0, 206.11111111111111], [388.66666666666663, 243.88888888888889],
          [433.1111111111111, 182.77777777777777], [457.55555555555554, 93.88888888888889],
          [480.8888888888889, 2.7777777777777777], [512.0, 6.111111111111111],
          [536.4444444444445, 82.77777777777777], [596.4444444444445, 168.33333333333334],
          [683.1111111111111, 171.66666666666666], [746.4444444444443, 227.22222222222223],
          [794.2222222222222, 296.1111111111111], [850.8888888888889, 359.44444444444446],
          [837.5555555555554, 413.88888888888886], [840.8888888888889, 465.0], [792.0, 519.4444444444445],
          [867.5555555555554, 686.1111111111111], [899.7777777777778, 1010.5555555555555],
          [879.7777777777778, 1076.111111111111]]

process_points_fill(im_path='/Users/sherry/work/pycharm_python/generator_tfrecord_data/timg.jpg', points=points,
                    output_path='./')
