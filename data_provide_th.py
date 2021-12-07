import threading
import time
import os
from tqdm import tqdm
import numpy as np
import gen_and_get_lable_set
import reshape_data_224
import re
try:
    import queue
except ImportError:
    import Queue as queue


lable_txt = None
re_str = r'$file_name.jpg (.+)\n'
lable_dic = None
lable_dic_opposite = None


def load_lable(lable_path, file_name):
    global lable_txt
    global lable_dic
    if lable_txt is None:
        with open(lable_path, "r") as f:
            lable_txt = f.read()
    if lable_dic is None:
        lable_dic = gen_and_get_lable_set.read_lable_dict()

    lable_group = re.compile(re_str.replace("$file_name", file_name)).search(lable_txt)
    if lable_group is not None:
        lable = lable_group.group(1)
        lable = lable.replace(" ", "")
    else:
        lable = ""

    if lable != "":
        lable_idx = [int(lable_dic[_]) for _ in list(lable)]
    else:
        lable_idx = []
    return lable_idx


class data_producer(object):
    def __init__(self, img_dir, label_dir):
        self.q = queue.Queue(maxsize=2)  # 在里面设置队列的大小
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_path = "image"
        self.label_path = "label"

    def producer_image(self):
        im_fns = os.listdir(os.path.join(self.img_dir, self.img_path))  # 图片目录
        for im_fn in im_fns:
            _, fn = os.path.split(im_fn)
            bfn, ext = os.path.splitext(fn)  # 将文件名和扩展名分开
            if ext.lower() not in ['.jpg', '.png']:
                continue
            labels_path = os.path.join(self.label_dir, self.label_path, '' + '360_train' + '.txt')
            img_path = os.path.join(self.img_dir, self.img_path, im_fn)
            # 原始文件的方式转换数据
            img, img_w, img_shape = reshape_data_224.reshape_picture_224_for_th(img_path)
            img = img * (1. / 255) - 0.5
            img = np.expand_dims(img, axis=0)
            img_w = np.expand_dims(img_w, axis=0)
            lable_idx = load_lable(labels_path, bfn)
            lable_idx = np.expand_dims(lable_idx, axis=0)
            lable_seq_len = np.expand_dims(len(lable_idx), axis=0)
            img_shape = np.expand_dims(img_shape, axis=0)
            if img is not None and len(lable_idx) > 0:
                feature = {"img_data": img,
                           "img_inf": img_shape,
                           "img_width": img_w,
                           "lables": lable_idx,
                           "lable_seq_len": lable_seq_len}
                self.q.put(feature)
                print("put......")
            # time.sleep(1)

    def get_image(self):
        return self.q.get()


if __name__ == '__main__':
    dp = data_producer("/Users/sherry/data/Synthetic_Chinese_String_Dataset_part",
                       "/Users/sherry/data/Synthetic_Chinese_String_Dataset_label")
    p = threading.Thread(dp.producer_image())
    g = threading.Thread(dp.get_image())
    p.start()
    p.join()
    while True:
        print("111")
        train_data = dp.get_image()


