import random
import os

import tensorflow as tf
import numpy as np
import matplotlib.image as iming
from PIL import Image
from cls_and_cnn.one_hott import *

cnn_train = '/Users/wywy/Desktop/cnn_train_2'
cnn_train_filename = './cnn_train.tfrecords_2'  # 输出文件地址


def saver_lables(lables_path,train_filename):
    writer = tf.python_io.TFRecordWriter(train_filename)
    all_filename=[]
    for i in os.listdir(cnn_train):
        if i=='.DS_Store':
            pass
        else:
            all_filename.append(i)
    random.shuffle(all_filename)
    for ii in all_filename:
        sample=ii.split('.')[0].split('_')[1]
        if sample=='A':
            sample=0.
        elif sample=='B':
            sample=1.
        elif sample=='C':
            sample=2.
        elif sample=='D':
            sample=3.
        elif sample=='X':
            sample=4.
        else:
            pass
        img=Image.open(cnn_train+'/'+ii)
        image = img.tobytes()

        example = tf.train.Example(features=tf.train.Features(feature={
                'lables': tf.train.Feature(float_list=tf.train.FloatList(value=[sample])),
                'images': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image]))
            }))
        writer.write(example.SerializeToString())  # 序列化为字符串
    writer.close()


def read_data_for_file(file, capacity,image_size):
    # 创建文件队列,不限读取的数量
    filename_queue = tf.train.string_input_producer([file], shuffle=True, capacity=capacity)

    # create a reader from file queue
    reader = tf.TFRecordReader()
    # reader从文件队列中读入一个序列化的样本
    _, serialized_example = reader.read(filename_queue)

    # 解析符号化的样本
    features = tf.parse_single_example(
        serialized_example,
        features={
            # 'class_lables': tf.FixedLenFeature([], tf.float32),
            'lables': tf.FixedLenFeature([], tf.float32),
            'images':tf.FixedLenFeature([1], tf.string)
        }
    )
    img = tf.decode_raw(features['images'], tf.uint8)
    img=tf.reshape(img,image_size)

    img = tf.cast(img, tf.float32) / 255.-0.5
    lables = tf.cast(features['lables'], tf.int32)
    lables=tf.reshape(lables,[1])


    return img, lables


def cnn_train_shuffle_batch(train_file_path, image_size,batch_size, capacity=7000, num_threads=3):
    images, lables = read_data_for_file(train_file_path, 10000,image_size)

    images_,  lables_ = tf.train.shuffle_batch([images,lables], batch_size=batch_size, capacity=capacity,
                                               min_after_dequeue=1000,
                                               num_threads=num_threads)
    return images_, lables_



# init = tf.global_variables_initializer()
if __name__=='__main__':
    saver_lables(cnn_train,cnn_train_filename)
    # read_data_for_file(train_filename,100,[16,88,3])
    a=cnn_train_shuffle_batch(cnn_train_filename,[18,88,3],100)
    with tf.Session() as sess:
        coord = tf.train.Coordinator()  # 创建一个协调器，管理线程
        threads = tf.train.start_queue_runners(coord=coord,sess=sess)

        # sess.run(init)

        aa,bb,=sess.run(a)
        print(aa.shape,
              bb
          )