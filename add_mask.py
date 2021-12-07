# coding=utf-8

import cv2
import sys
import random

im_path = "/Users/sherry/data/Synthetic_Chinese_String_Dataset_part/image/20436328_800384098.jpg"
im = cv2.imread(im_path)
print(im.shape)
en = False  # 使能，鼠标左键开启


# 鼠标事件
def draw(event, x, y, flags, param):
    global en
    if event == cv2.EVENT_LBUTTONDOWN:
        en = True  # 使能开启
    elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_LBUTTONDOWN:
        if en:
            drawMask(y, x)  # 强行打码
        elif event == cv2.EVENT_LBUTTONUP:
            en = False


# 打码函数
def drawMask(x=10, y=50, size=10):
    # 为了让码好看一些,做了一个size*size的分区处理
    X = int(x / size * size)
    Y = int(y / size * size)
    print(X, Y, im.shape[2])
    # for z in range(im.shape[2]):
    for i in range(size):
        for j in range(size):
            im[X + i][Y + j] = im[X][Y]
            im[X + i][Y + j] = random.randint(0, 255)
            # im[X+i][Y+j][z]=im[X][Y][z]


cv2.namedWindow('image')
cv2.setMouseCallback('image', draw)

while (1):
    drawMask()
    cv2.imshow('image', im)
    cv2.waitKey(0)
    '''
    if cv2.waitKey(10)&0xFF==27: #‘esc’退出
        break
    elif cv2.waitKey(10)&0xFF==115:#‘s’键保存图片
        cv2.imwrite(save_path.encode('gbk'),im)
    '''
cv2.destroyAllWindows()
