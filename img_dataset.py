import tensorflow as tf
import math
import numpy as np
import img_dataset_util
from tensorflow.python.framework import dtypes
import cv2 as cv


"""
    使用 Dataset api 并行读取图片数据
    参考：
        - 关于 TF Dataset api 的改进讨论：https://github.com/tensorflow/tensorflow/issues/7951
        - https://kratzert.github.io/2017/06/15/example-of-tensorflows-new-input-pipeline.html
        - https://stackoverflow.com/questions/47064693/tensorflow-data-api-prefetch
        - https://stackoverflow.com/questions/46444018/meaning-of-buffer-size-in-dataset-map-dataset-prefetch-and-dataset-shuffle

    TL;DR
        Dataset.shuffle() 的 buffer_size 参数影响数据的随机性， TF 会先取 buffer_size 个数据放入 catch 中，再从里面选取
        batch_size 个数据，所以使用 shuffle 有两种方法：
            1. 每次调用 Dataset api 前手动 shuffle 一下 filepaths 和 labels
            2. Dataset.shuffle() 的 buffer_size 直接设为 len(filepaths)。这种做法要保证 shuffle() 函数在 map、batch 前调用

        Dataset.prefetch() 的 buffer_size 参数可以提高数据预加载性能，但是它比 tf.FIFOQueue 简单很多。
        tf.FIFOQueue supports multiple concurrent producers and consumers
"""

resize_map = {0: cv.INTER_NEAREST,
              1: cv.INTER_LINEAR,
              2: cv.INTER_AREA,
              3: cv.INTER_CUBIC,
              4: cv.INTER_LANCZOS4}


# noinspection PyMethodMayBeStatic
class ImgDataset:
    """
    Use tensorflow Dataset api to load images in parallel
    """

    def __init__(self, img_dir, label_dir,
                 converter,
                 batch_size,
                 num_parallel_calls=4,
                 img_channels=3,
                 shuffle=True):
        self.converter = converter
        self.batch_size = batch_size
        self.num_parallel_calls = num_parallel_calls
        self.img_channels = img_channels
        self.img_dir = img_dir
        self.shuffle = shuffle
        img_paths, labels, lab_idx = img_dataset_util.load_image_and_labels(img_dir, label_dir)
        self.size = len(img_paths)
        dataset = self._create_dataset(img_paths, lab_idx)
        iterator = tf.data.Iterator.from_structure(dataset.output_types,
                                                   dataset.output_shapes)
        self.next_batch = iterator.get_next()
        self.init_op = iterator.make_initializer(dataset)

        self.num_batches = math.ceil(self.size / self.batch_size)


    def get_next_batch(self, sess):
        """return images and labels of a batch"""
        img_batch, labels, img_paths = sess.run(self.next_batch)
        img_paths = [x.decode() for x in img_paths]
        labels = [l.decode() for l in labels]

        encoded_label_batch = self.converter.encode_list(labels)
        sparse_label_batch = self._sparse_tuple_from_label(encoded_label_batch)
        return img_batch, sparse_label_batch, labels, img_paths

    def _sparse_tuple_from_label(self, sequences, default_val=-1, dtype=np.int32):
        """Create a sparse representention of x.
        Args:
            sequences: a list of lists of type dtype where each element is a sequence
                      encode label, e.g: [2,44,11,55]
            default_val: value should be ignored in sequences
        Returns:
            A tuple with (indices, values, shape)
        """
        indices = []
        values = []

        for n, seq in enumerate(sequences):
            seq_filtered = list(filter(lambda x: x != default_val, seq))
            indices.extend(zip([n] * len(seq_filtered), range(len(seq_filtered))))
            values.extend(seq_filtered)

        indices = np.asarray(indices, dtype=np.int64)
        values = np.asarray(values, dtype=dtype)

        if len(indices) == 0:
            shape = np.asarray([len(sequences), 0], dtype=np.int64)
        else:
            shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

        return indices, values, shape

    def _create_dataset(self, img_paths, labels):
        img_paths = tf.convert_to_tensor(img_paths, dtype=dtypes.string)
        labels = tf.convert_to_tensor(labels, dtype=dtypes.int32)

        d = tf.data.Dataset.from_tensor_slices((img_paths, labels))
        if self.shuffle:
            d = d.shuffle(buffer_size=self.size)

        d = d.map(self._input_parser,
                  num_parallel_calls=self.num_parallel_calls)
        d = d.batch(self.batch_size)
        # d = d.repeat(self.num_epochs)
        d = d.prefetch(buffer_size=2)
        return d

    def _input_parser(self, img_path, label, img_h=32, img_w=280):  # train data shape is (32, 280)
        """
        img_file = tf.read_file(img_path)
        img_decoded = tf.image.decode_jpeg(img_file, channels=self.img_channels)
        img_decoded = tf.image.resize_images(
            images=img_decoded, size=[int(img_h/2), int(img_w/2)], method=np.random.randint(0, 4))
        img_decoded = tf.image.resize_images(
            images=img_decoded, size=[img_h*7, img_w*7], method=np.random.randint(0, 4))

        lable_seq_len = 10
        return img_decoded, img_w*7, [img_h*7, img_w*7, self.img_channels], label, lable_seq_len
        """

        img_s, img_w, img_shape = ImgDataset.reshape_picture_224_for_th(img_path)
        img = img_s * (1. / 255)
        lable_seq_len = len(label)
        return img, img_w, img_shape, label, lable_seq_len
        

    @staticmethod
    def reshape_picture_224_for_th(img_path):
        img = cv.imread(img_path)  # img is numpy.array
        if img is not None:
            img = ImgDataset.gaussian_noise_demo(img)  # 添加噪声
            x, y = img.shape[0:2]
            r_x = 16. / x
            img_16 = cv.resize(img, (0, 0), fx=r_x, fy=r_x,
                               interpolation=resize_map[np.random.randint(0, 5)])
            r_x = 224. / 16.
            img_224 = cv.resize(img_16, (0, 0), fx=r_x, fy=r_x,
                                interpolation=resize_map[np.random.randint(0, 5)])
            return img_224, img_224.shape[1], img_224.shape

    @staticmethod
    def gaussian_noise_demo(image):
        """添加高斯噪声"""
        h, w, c = image.shape
        for row in range(0, h):
            for col in range(0, w):
                s = np.random.normal(0, 15, 3)  # 产生随机数，每次产生三个
                b = image[row, col, 0]  # blue
                g = image[row, col, 1]  # green
                r = image[row, col, 2]  # red
                image[row, col, 0] = ImgDataset.clamp(b + s[0])
                image[row, col, 1] = ImgDataset.clamp(g + s[1])
                image[row, col, 2] = ImgDataset.clamp(r + s[2])
        return image

    @staticmethod
    def clamp(pv):
        """防止溢出"""
        if pv > 255:
            return 255
        elif pv < 0:
            return 0
        else:
            return pv

    """
    img = gaussian_noise_demo(img)  # 添加噪声
        x, y = img.shape[0:2]
        r_x = 16. / x
        img_16 = cv.resize(img, (0, 0), fx=r_x, fy=r_x,
                           interpolation=resize_map[np.random.randint(0, 5)])
        r_x = 224. / 16.
        img_224 = cv.resize(img_16, (0, 0), fx=r_x, fy=r_x,
                            interpolation=resize_map[np.random.randint(0, 5)])
        return img_224, img_224.shape[1], img_224.shape
    """

    def _input_parser_bak(self, img_path, label):
        img_file = tf.read_file(img_path)

        img_decoded = tf.image.decode_image(img_file, channels=self.img_channels)
        if self.img_channels == 3:
            img_decoded = tf.image.rgb_to_grayscale(img_decoded)

        img_decoded = tf.cast(img_decoded, tf.float32)
        img_decoded = (img_decoded - 128.0) / 128.0

        return img_decoded, label, img_path


if __name__ == '__main__':
    from label_converter import LabelConverter

    img_path = "/Users/sherry/data/Synthetic_Chinese_String_Dataset_part"
    chars_file = "/Users/sherry/data/Synthetic_Chinese_String_Dataset_label"

    epochs = 5
    batch_size = 2

    converter = LabelConverter(chars_file=chars_file)
    ds = ImgDataset(img_path, chars_file, converter, batch_size=batch_size)

    num_batches = int(np.floor(ds.size / batch_size))

    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        for epoch in range(epochs):
            sess.run(ds.init_op)
            print('------------Epoch(%d)------------' % epoch)
            for batch in range(num_batches):
                img_batch, sparse_label_batch, labels, img_paths = ds.get_next_batch(sess)
                print(img_batch.shape)
                print(sparse_label_batch.shape)
                print(labels.shape)
                print(img_paths.shape)

