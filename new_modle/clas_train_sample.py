import random
import os

import tensorflow as tf
import numpy as np
import matplotlib.image as iming
from PIL import Image
from cls_and_cnn.one_hott import *

train_path='/Users/wywy/Desktop/客观题分类/cls_train'
train_filename = './cls_train.tfrecords'  # 输出文件地址


def saver_lables(lable_all,train_filename):
    writer = tf.python_io.TFRecordWriter(train_filename)
    all_path=[]
    for i in os.listdir(lable_all):
        if i == '.DS_Store':
            os.remove(lable_all + '/' + i)
        else:
            all_path.append(lable_all+'/'+i)

    random.shuffle(all_path)
    for j in all_path:
        lable=float(j.split('/')[-1].split('.')[0].split('_')[-1])
        img=Image.open(j).convert('L')
        img=img.resize((168, 16), Image.ANTIALIAS)
        image = img.tobytes()
        img_name=str.encode(j)

        example = tf.train.Example(features=tf.train.Features(feature={
                'lables': tf.train.Feature(float_list=tf.train.FloatList(value=[lable])),
                'images': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
                'img_filename':tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_name]))
            }))
        writer.write(example.SerializeToString())  # 序列化为字符串
    writer.close()


def read_data_for_file(file, capacity,image_size):
    # 创建文件队列,不限读取的数量
    filename_queue = tf.train.string_input_producer([file], num_epochs=None,shuffle=True, capacity=capacity)
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
            'images':tf.FixedLenFeature([], tf.string),
            'img_filename':tf.FixedLenFeature([], tf.string)
        }
    )
    img = tf.decode_raw(features['images'], tf.uint8)
    img=tf.reshape(img,image_size)

    img = tf.cast(img, tf.float32)
    lables = tf.cast(features['lables'], tf.int32)
    lables=tf.reshape(lables,[1])
    file_name = tf.cast(features['img_filename'], tf.string)


    return img, lables,file_name


def train_shuffle_batch(train_file_path, image_size,batch_size, capacity=700, num_threads=3):
    images, lables,file_name = read_data_for_file(train_file_path, 1000,image_size)

    images_,  lables_,file_name_ = tf.train.shuffle_batch([images,lables,file_name], batch_size=batch_size, capacity=capacity,
                                               min_after_dequeue=100,
                                               num_threads=num_threads)
    return images_, lables_,file_name_



init = tf.global_variables_initializer()
if __name__=='__main__':
    saver_lables(train_path,train_filename)
    a=train_shuffle_batch(train_filename,[16,168,1],100)
    with tf.Session() as sess:
        coord = tf.train.Coordinator()  # 创建一个协调器，管理线程
        threads = tf.train.start_queue_runners(coord=coord,sess=sess)

        sess.run(init)
        for ii in range(50):
            aa,bb,cc=sess.run(a)
            print(bytes.decode(cc[0]))