import random
import os
import cv2 as cv

import tensorflow as tf

from PIL import Image


train_img_path = '/Users/wywy/Desktop/cls_train'
train_filename = './cls_train.tfrecords'  # 输出文件地址


def saver_lables(img_path,train_filename):
    writer = tf.python_io.TFRecordWriter(train_filename)
    all_filename = []
    for i in os.listdir(img_path):
        if i == '.DS_Store':
            os.remove(img_path + '/' + i)
        else:
            all_filename.append(i)
    random.shuffle(all_filename)
    for ii in all_filename:
        sample = float(ii.split('.')[0].split('_')[-1])
        img = Image.open(img_path + '/' + ii)
        img=img.convert('RGB')
        img=img.resize((168,16),Image.ANTIALIAS)
        image = img.tobytes()

        example = tf.train.Example(features=tf.train.Features(feature={
            'lables': tf.train.Feature(float_list=tf.train.FloatList(value=[sample])),
            'images': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
            'file_name': tf.train.Feature(bytes_list=tf.train.BytesList(value=[str.encode(ii)]))
        }))

        writer.write(example.SerializeToString())  # 序列化为字符串
    writer.close()


def read_data_for_file(file, capacity,image_size):
    # 创建文件队列,不限读取的数量
    filename_queue = tf.train.string_input_producer([file], num_epochs=None, shuffle=False, capacity=capacity)

    # create a reader from file queue
    reader = tf.TFRecordReader()
    # reader从文件队列中读入一个序列化的样本
    _, serialized_example = reader.read(filename_queue)

    # 解析符号化的样本
    features = tf.parse_single_example(
        serialized_example,
        features={
            'lables': tf.FixedLenFeature([], tf.float32),
            'images':tf.FixedLenFeature([], tf.string),
            'file_name':tf.FixedLenFeature([], tf.string)

        }
    )
    img = tf.decode_raw(features['images'], tf.uint8)
    img=tf.reshape(img,image_size)
    img = tf.cast(img, tf.float32)
    setoff_lables=features['lables']
    file_name= tf.cast(features['file_name'], tf.string)


    return img, setoff_lables,file_name
#
def train_shuffle_batch(train_file_path,image_size, batch_size, capacity=7000, num_threads=3):
    images, setoff_lables,file_name = read_data_for_file(train_file_path, 10000,image_size)

    images_,  setoff_lables_,file_name_ = tf.train.shuffle_batch([images,setoff_lables,file_name], batch_size=batch_size, capacity=capacity,
                                               min_after_dequeue=1000,
                                               num_threads=num_threads)
    return images_, setoff_lables_,file_name_




if __name__=='__main__':
    saver_lables(train_img_path,train_filename)
    # read_data_for_file(train_filename,100,[16,88,3])
    a=train_shuffle_batch(train_filename,[16,168,3],1000)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        coord = tf.train.Coordinator()  # 创建一个协调器，管理线程
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        sess.run(init)

        aa, bb ,cc= sess.run(a)
        print(aa[0].shape)
        print(bb[0])

