# coding=utf-8
import cv2 as cv
import numpy as np


def read_img_and_compare(file_parh):
    img = cv.imread(file_parh)  # img is numpy.array
    img = img * (1. / 255) - 0.5
    print(img.shape)
    img = np.expand_dims(img, axis=0)
    print(type(img), img.shape)


if __name__ == '__main__':
    read_img_and_compare("/Users/sherry/data/Synthetic_Chinese_String_Dataset_224/image/20436578_2907107908.jpg")
