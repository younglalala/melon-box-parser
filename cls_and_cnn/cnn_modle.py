import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
from cls_and_cnn.cnn_test1_sample import *
from cls_and_cnn.cnn_test2_sample import *
from cls_and_cnn.cnn_train_sample import *
tf.set_random_seed(1)

class ClsCnn:
    def __init__(self):
        self.x=tf.placeholder(tf.float32,[None,18,88,3],name='input')   #图片大小自定义，对图片进行归一化处理。
        self.y_=tf.placeholder(tf.float32,[None,5])
        # self.dp=tf.placeholder(tf.float32)

        self.conv1_w = tf.Variable(tf.random_normal([3, 3, 3, 10], dtype=tf.float32, stddev=tf.sqrt(1/10)))
        self.conv1_b = tf.Variable(tf.zeros([10]))

        self.conv2_w = tf.Variable(tf.random_normal([5, 5, 10, 16], dtype=tf.float32, stddev=tf.sqrt(1/16)))
        self.conv2_b = tf.Variable(tf.zeros([16]))

        self.fc1=tf.Variable(tf.random_normal([2*7*16,128],dtype=tf.float32,stddev=tf.sqrt(1/128)))
        self.b1=tf.Variable(tf.zeros([128]))
        self.fc2=tf.Variable(tf.random_normal([128,5],dtype=tf.float32,stddev=tf.sqrt(1/5)))
        self.b2 = tf.Variable(tf.zeros([5]))

    def forward(self):
        self.conv1=tf.nn.leaky_relu(tf.layers.batch_normalization( tf.nn.conv2d(self.x,self.conv1_w,strides=[1,2,2,1],padding='SAME')+self.conv1_b))   #9,44,10
        self.pool1=tf.nn.max_pool(self.conv1, ksize=[1, 3, 3, 1],strides=[1, 3, 3, 1], padding='VALID')  #3,14,10
        self.conv2 = tf.nn.leaky_relu(tf.layers.batch_normalization(tf.nn.conv2d(self.pool1, self.conv2_w, strides=[1, 2, 2, 1], padding='SAME') + self.conv2_b) )  #2,7,16

        flat = tf.reshape(self.conv2, [-1, 2 * 7 * 16])

        self.y1 = tf.nn.leaky_relu(tf.layers.batch_normalization(tf.matmul(flat, self.fc1) + self.b1))
        # self.y11 = tf.nn.dropout(self.y1, keep_prob=self.dp)
        self.out1=tf.matmul(self.y1, self.fc2) + self.b2
        self.out=tf.reshape(self.out1,[-1,5])

        return self.out


    def backward(self):
        # self.loss = tf.reduce_mean(tf.pow(self.out-self.y_,2) ) # 求误差
        self.loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.out,labels=self.y_))

        self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.loss)
        self.correct_prediction = tf.equal(tf.argmax(self.out, 1), tf.argmax(self.y_, 1))
        self.rst = tf.cast(self.correct_prediction, "float")
        self.accuracy = tf.reduce_mean(self.rst)
        self.out_argmax=tf.argmax(self.out, 1)
        self.out_argmax1=tf.reshape(self.out_argmax,[-1],name='output')


if __name__=='__main__':

    cnn_net = ClsCnn()
    cnn_net.forward()
    cnn_net.backward()

    train_data = cnn_train_shuffle_batch(cnn_train_filename, [18, 88, 3],100)
    test1_data = cnn_test1_shuffle_batch(cnn_test1_filename,[18,88,3],1000)
    test2_data = cnn_test2_shuffle_batch(cnn_test2_filename, [18, 88, 3], 100)

    init = tf.global_variables_initializer()
    x=[]
    y=[]
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        coord = tf.train.Coordinator()    # 创建一个协调器，管理线程
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        # saver.restore(sess, "modle_save/cnn_modle.dpk")
        for ii in range(10000):

            imgs, lables= sess.run(train_data)
            test1_img,test1_lable=sess.run(test1_data)
            test2_img,test2_lable=sess.run(test2_data)

            lables=one_hot(lables.reshape([-1]))
            test1_lables=one_hot(test1_lable.reshape([-1]))
            test2_lable=one_hot(test2_lable.reshape([-1]))

            out_argmax1,_,loss,out,accuracy=sess.run([cnn_net.out_argmax1, cnn_net.optimizer,cnn_net.loss,cnn_net.out,cnn_net.accuracy],
                                     feed_dict={cnn_net.x:imgs,cnn_net.y_:lables})

            print('第{}次的误差为{}********精度为{},out_argmax1:{}'.format(ii,loss,accuracy,out_argmax1))
            if ii%100==0:
                #
                test_loss, outtest,ac = sess.run([cnn_net.loss, cnn_net.out,cnn_net.accuracy],
                                                  feed_dict={cnn_net.x: test2_img, cnn_net.y_: test2_lable})

                outset_dix=np.where(np.max(outtest,0))
                print(outset_dix,'=======================')
                print('测试集的loss{}，输{} ，jingdu{}'.format(test_loss,test1_lable,ac))

                saver.save(sess, "modle_save/cnn_modle.dpk")
                graph_def = tf.get_default_graph().as_graph_def()
                output_graph_def = tf.graph_util.convert_variables_to_constants(sess, graph_def,
                                                                                ['output'])

                with tf.gfile.GFile("./cnn_modle.pb", 'wb') as f:
                    f.write(output_graph_def.SerializeToString())


            y.append(loss)
            x.append(ii)
            plt.plot(x,y,'red')
            plt.pause(0.1)
            plt.clf()









