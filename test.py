import os
import re
import numpy as np
import tensorflow as tf
import random

lable_dir = r'/Users/sherry/data/Synthetic Chinese String Dataset label'

print(os.path.join(lable_dir, "label", '' + "360_train" + '.txt'))

'''
with open("/Users/sherry/data/Synthetic_Chinese_String_Dataset_label/label/360_train.txt", 'r', encoding='utf-8') as f:
    text = f.read()
print(text)
'''

re_str = r'72629765_3424604982.jpg (.+)\n'

res = re.compile(re_str).search("72629765_3424604982.jpg 文明dd程度已经翻了几番\n")
print(res.group(1))
print(re_str.replace("72629765_3424604982", ""))
print(re_str)

lable_index = list(range(100))
print(lable_index)

print(list("文明dd程度已经翻了几番"))

print(np.array([2, 3, 4]).reshape([1, 3]))

x = [[1, 2, 3, 4], [1, 2, 3, 4]]
print(x)
print(np.array(x))

print(os.path.join(lable_dir, "image"))


file_list = tf.train.match_filenames_once(
        "/Users/sherry/work/pycharm_python/generator_tfrecord_data/data/*.tfrecord")
init = (tf.global_variables_initializer(), tf.local_variables_initializer())
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(file_list))

im_fns = os.listdir("/Users/sherry/work/pycharm_python/generator_tfrecord_data/data")
im_fns = ["/Users/sherry/work/pycharm_python/generator_tfrecord_data/data" + os.sep + _ for _ in im_fns]
print(type(im_fns), im_fns)

list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
for i in range(3):
    slice = random.sample(list, 5)  # 从list中随机获取5个元素，作为一个片断返回
    print(type(slice), slice)
    print(list, '\n')  # 原有序列并没有改变

