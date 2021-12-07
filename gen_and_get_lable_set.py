# coding=utf-8
import os

lable_dic_file = "lable_set.txt"
lable_path = r"/Users/sherry/data/Synthetic_Chinese_String_Dataset_label"
lable_paths = os.listdir(lable_path)
ocr_dict = set()


def read_line(path):
    global ocr_dict
    f = open(path)  # 返回一个文件对象
    line = f.readline()  # 调用文件的 readline()方法
    while line:
        # print(line)  # 在 Python 3 中使用
        string = line.split(" ")[-1]
        ocr_dict = ocr_dict.union(set(string))
        # print(ocr_dict)
        line = f.readline()
    f.close()


def read_all_file(paths):
    for file in paths:
        print(file)
        read_line(os.path.join(lable_path, file))


def gen_dict():
    read_all_file(lable_paths)
    print(os.path.join(lable_path, "lable_set.txt"))
    with open(os.path.join(lable_path, "lable_set.txt"), 'w') as file_object:
        for _ in ocr_dict:
            print(_)
            file_object.write(_ + '\n')


def read_lable_dict():
    with open(os.path.join(lable_path, "lable_set.txt"), 'r', encoding='utf-8') as f:
        text = f.read()
    lable_list = text.split("\n")
    lable_index = list(range(len(lable_list)))
    lable_set = dict(zip(lable_list, lable_index))
    return lable_set


def read_lable_dict_opposite():
    with open(os.path.join(lable_path, "lable_set.txt"), 'r', encoding='utf-8') as f:
        text = f.read()
    lable_list = text.split("\n")
    lable_index = list(range(len(lable_list)))
    lable_set = dict(zip(lable_index, lable_list))
    return lable_set


if __name__ == '__main__':
    # gen_dict()
    lable_set_ = read_lable_dict()
    print(lable_set_)
    print(len(lable_set_))
