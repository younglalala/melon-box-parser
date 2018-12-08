from skimage import io ,data
import math
import  matplotlib.image as img

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from cls_and_cnn.cls_train_sample import *
from cls_and_cnn.cls_test1_sample import *
from cls_and_cnn.cls_test2_sample import *

tf.set_random_seed(1)

class Cls:
    def __init__(self):
        #
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, 18, 88, 3],name='cls_input')
        self.y_ = tf.placeholder(shape=[None, 1], dtype=tf.float32,name='cls_y')
        # self.dp = tf.placeholder(tf.float32,name='cls_dp')
        self.conv1_w = tf.Variable(tf.random_normal([3, 3, 3, 10], dtype=tf.float32, stddev=tf.sqrt(1 / 10)))
        self.conv1_b=tf.Variable(tf.zeros([10]))

        self.w1=tf.Variable(tf.random_normal([3*14*10,256],dtype=tf.float32, stddev=tf.sqrt(1 / 256)))
        self.b1=tf.Variable(tf.zeros([256]))

        self.w3 = tf.Variable(tf.random_normal([256, 1], dtype=tf.float32, stddev=tf.sqrt(1 / 1)))
        self.b3 = tf.Variable(tf.zeros([1]))
    def forward(self):
        self.conv1=tf.nn.leaky_relu(tf.layers.batch_normalization( tf.nn.conv2d(self.x,self.conv1_w,strides=[1,2,2,1],padding="SAME")+self.conv1_b))   #9,44,10

        self.pool1 = tf.nn.max_pool(self.conv1, ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1], padding='VALID')   #3,14,10

        self.flat=tf.reshape(self.pool1,[-1,3*14*10])
        self.y1=tf.nn.leaky_relu(tf.layers.batch_normalization(tf.matmul(self.flat,self.w1)+self.b1))
        # self.y_dp = tf.nn.dropout(self.y1, keep_prob=self.dp)                     #加dropout  防止过拟合。

        self.out11=tf.nn.sigmoid(tf.matmul(self.y1,self.w3)+self.b3)
        self.out=tf.reshape(self.out11,[-1,1],name='cls_output')

        return self.out


    def bacwrad(self):
        self.loss=tf.reduce_mean(tf.pow(self.out-self.y_,2)+0.00001*tf.reduce_sum(tf.pow(self.conv1_w,2))
                                 +0.00001 * tf.reduce_sum(tf.pow(self.w1, 2))
                                 +0.00001*tf.reduce_sum(tf.pow(self.w3,2)))       #加正则化，限制权重，防止过拟合
        self.optimizer=tf.train.AdamOptimizer(0.0001).minimize(self.loss)        #学习率尽可能小一点，以便于找到最优点。


if __name__=='__main__':
    cls=Cls()
    cls.forward()
    cls.bacwrad()
    dataset = train_shuffle_batch(train_filename, [18, 88, 3],100)
    test1_data=test1_shuffle_batch(test1_filename,[18,88,3],1000)
    test2_data = test2_shuffle_batch(test2_filename, [18, 88, 3],5000)

    init=tf.global_variables_initializer()
    x = []
    y = []
    x1 = []
    y1 = []
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        coord = tf.train.Coordinator()  # 创建一个协调器，管理线程
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        saver.restore(sess, "modle_save/scls_modle.dpk")
        plt.ion()
        for i in range(30000):
            imgs, lables = sess.run(dataset)

            _, loss, out= sess.run(
                [cls.optimizer, cls.loss, cls.out],
                feed_dict={cls.x: imgs, cls.y_: lables})
            print('第{}次的误差是{}，输入是{}，输出{}'.format(i, loss, lables[0], out[0]))
            x.append(i)
            y.append(loss)
            plt.plot(x,y,'red')
            plt.pause(0.5)
            plt.clf()
            if i%100==0:
                saver.save(sess, "modle_save/scls_modle.dpk")
                graph_def = tf.get_default_graph().as_graph_def()
                output_graph_def = tf.graph_util.convert_variables_to_constants(sess, graph_def,
                                                                                ['cls_output'])

                with tf.gfile.GFile("./cls_modle.pb", 'wb') as f:
                    f.write(output_graph_def.SerializeToString())
            #
            # print('第{}次的误差是{}，输入是{}，输出{}'.format(i,loss,lables[0],out[0]))
            #
            # # bad_case_cls=np.less_equal(out,0.1)
            # # bad_case_idx=np.where(bad_case_cls)
            # # list_badcase=bad_case_idx[0].tolist()
            # # bad_case_img=(imgs[list_badcase])
            #
            #
            if i%10==0:
                test1_img, test1_lable = sess.run(test1_data)
                test2_img, test2_lable = sess.run(test2_data)
                testout1=sess.run(cls.out,feed_dict={cls.x: test2_img, cls.y_: test2_lable})
                print('测试集111集第{}，输入是{}，输出{}'.format(i,test2_lable[0],testout1[0]))
                print(testout1)


                bad_case_cls=np.less_equal(testout1,0.1)
                bad_case_idx=np.where(bad_case_cls)

                good_lablecls=np.greater(testout1,0.1)
                good_idx = np.where(good_lablecls)
                good_idx_list=good_idx[0].tolist()
                good_img=(test2_img[good_idx_list])

                test_lable_bad=np.less_equal(test2_lable,0.1)
                bad_test_idx = np.where(test_lable_bad)
                bad_test_list=bad_test_idx[0].tolist()

                list_badcase=bad_case_idx[0].tolist()
                bad_case_img=(test2_img[list_badcase])
                no_find1= list(set(list_badcase).difference(bad_test_list))
                print(no_find1,'no_find1')
                print(good_img.shape)
                if len(bad_case_img)==0:
                    pass
                else:
                    plt.subplot(313)
                    plt.title('iamge')
                    for n in range(len(bad_case_img)):
                        plt.imshow(bad_case_img[n])
                        plt.pause(1)

            y.append(loss)
            x.append(i)
            # plt.subplot(211)
            plt.title('train')
            plt.plot(x, y, 'red')
            plt.pause(0.1)

            # plt.show()




