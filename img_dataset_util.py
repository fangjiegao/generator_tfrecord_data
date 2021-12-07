# encoding:utf-8
import os
import re
import numpy as np

IMG_FOLDER = "/Users/sherry/data/Synthetic_Chinese_String_Dataset_part"
LABEL_FOLDER = "/Users/sherry/data/Synthetic_Chinese_String_Dataset_label"
lable_txt = None
re_str = r'$file_name.jpg (.+)\n'
lable_dic = None
lable_dic_opposite = None
lable_dic_file = "lable_set.txt"


def load_lable(lable_path, file_name):
    global lable_txt
    global lable_dic
    if lable_txt is None:
        label_file = os.path.join(lable_path, "label", "360_train.txt")

        with open(label_file, "r", encoding='utf-8') as f:
            lable_txt = f.read()
    if lable_dic is None:
        lable_dic = read_lable_dict(lable_path)

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
    return lable_idx, lable


def read_lable_dict(lable_path):
    with open(os.path.join(lable_path, "lable_set.txt"), 'r', encoding='utf-8') as f:
        text = f.read()
    lable_list = text.split("\n")
    lable_index = list(range(len(lable_list)))
    lable_set = dict(zip(lable_list, lable_index))
    return lable_set


def get_training_data(img_dir):
    img_files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    # for parent, dirnames, filenames in os.walk(os.path.join(DATA_FOLDER, "image")):
    for parent, dirnames, filenames in os.walk(os.path.join(img_dir, "image")):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    img_files.append(os.path.join(parent, filename))
                    break
    print('Find {} images'.format(len(img_files)))
    return img_files


def load_image_and_labels(img_dir, label_dir):
    image_list = np.array(get_training_data(img_dir))
    print('{} training images in {}'.format(image_list.shape[0], img_dir))
    lable_list = []
    lable_idx_list = []
    for image_path in image_list:
        filepath, fullflname = os.path.split(image_path)
        fname, ext = os.path.splitext(fullflname)
        lable_idx, lable = load_lable(label_dir, fname)
        lable_list.append(lable)
        lable_idx_list.append(lable_idx)
    return image_list, lable_list, lable_idx_list


def load_chars(filepath):
    ret = ''
    with open(os.path.join(filepath, "lable_set.txt"), 'r', encoding='utf-8') as f:
        while True:
            line = f.readline()
            if not line:
                break
            ret += line.strip("\n")
    return ret


if __name__ == '__main__':
    img_list, lab_list, lab_idx_list = load_image_and_labels(IMG_FOLDER, LABEL_FOLDER)
    if len(img_list) == len(lab_list):
        for _ in range(len(img_list)):
            print(img_list[_])
            print(lab_list[_])
    else:
        print(len(img_list), len(lab_list))

    r = load_chars(LABEL_FOLDER)
    print(len(r))
