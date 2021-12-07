# coding: utf-8
import tensorflow as tf
import cv2

file1 = '/Users/sherry/data/Synthetic_Chinese_String_Dataset_part/image/20461687_2304302683.jpg'
file2 = "/Users/sherry/data/Synthetic_Chinese_String_Dataset_part/image/20461687_2304302683.jpg"
file_contents = tf.read_file(tf.convert_to_tensor(file1))
image = tf.image.decode_image(file_contents)  # 解码png格式
image = image.reshape(1, 32, 280, 3)
img_decoded = tf.image.resize_images(
            images=image, size=(224, 1960), method=0)
print(img_decoded.shape)

img2 = tf.image.decode_image(tf.read_file(file2))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    img, img2 = sess.run([img_decoded, img2])
    print(img.shape)
    print(img2.shape)
    cv2.imshow('file1', img[0])
    cv2.imshow('file2', img2)
    cv2.waitKey(0)
