import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


def test_gfile_read_img():
    # image_raw_data = tf.gfile.FastGFile(
    image_raw_data = tf.gfile.GFile(
        '/Users/sherry/data/Synthetic_Chinese_String_Dataset_224/image/20436578_2907107908.jpg', 'rb').read()

    with tf.Session() as sess:
        img_data = tf.image.decode_jpeg(image_raw_data)
        print(img_data.eval().shape)
        plt.imshow(img_data.eval())
        plt.show()

        img_data = tf.image.decode_jpeg(image_raw_data)
        img_data = tf.cast(img_data, tf.float32) * (1. / 255) - 0.5
        img_data_ = sess.run(img_data)
        print(img_data_.shape)
        print(img_data_)

        '''
        img_data = tf.image.decode_jpeg(image_raw_data)
        plt.imshow(img_data.eval())
        plt.show()
    
        resized = tf.image.resize_images(img_data, [100, 100], method=0)
        # TensorFlow的函数处理图片后存储的数据是float32格式的，需要转换成uint8才能正确打印图片。
        print("Digital type: ", resized.dtype)
        resized = np.asarray(resized.eval(), dtype='uint8')
        # tf.image.convert_image_dtype(rgb_image, tf.float32)
        plt.imshow(resized)
        plt.show()
    
        croped = tf.image.resize_image_with_crop_or_pad(img_data, 100, 100)
        padded = tf.image.resize_image_with_crop_or_pad(img_data, 500, 500)
        plt.imshow(croped.eval())
        plt.show()
        plt.imshow(padded.eval())
        plt.show()
    
        central_cropped = tf.image.central_crop(img_data, 0.5)
        plt.imshow(central_cropped.eval())
        plt.show()
        '''


def resize_img_by_tf(img_path):
    image_raw_data = tf.gfile.GFile(img_path, 'rb').read()

    with tf.Session() as sess:
        img_data = tf.image.decode_jpeg(image_raw_data)
        print(img_data.eval().shape)
        plt.imshow(img_data.eval())
        plt.show()

        img_data = tf.image.decode_jpeg(image_raw_data)
        img_data = tf.cast(img_data, tf.float32) * (1. / 255) - 0.5
        img_data_ = sess.run(img_data)
        print(img_data_.shape)
        print(img_data_)


if __name__ == '__main__':
    test_gfile_read_img()