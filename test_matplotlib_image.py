import numpy as np
import matplotlib.image as pm
import cv2 as cv
import tensorflow as tf
# 不同的读取图片的方式

img = pm.imread('/work/pycharm_python/text-detection-ctpn/text-detection-ctpn-banjin-dev/data/test/image/image_0.jpg')
print("matplotlib:", type(img))
print(isinstance(img, np.ndarray))

img = cv.imread('/work/pycharm_python/text-detection-ctpn/text-detection-ctpn-banjin-dev/data/test/image/image_0.jpg')
print("cv:", type(img))

img = tf.gfile.FastGFile(
    '/work/pycharm_python/text-detection-ctpn/text-detection-ctpn-banjin-dev/data/test/image/image_0.jpg', 'rb').read()
print("FastGFile:", type(img))
print(isinstance(img, bytes))
