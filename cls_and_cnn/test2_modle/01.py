
from cls_and_cnn.test2_modle.clas_train_sample import *
from cls_and_cnn.test2_modle.clas_test_sample import *

import matplotlib.pyplot as plt
import scipy.misc
import numpy as np


def one_hott(data):

    all_one_hot=[]
    for num in data:
        one_hot = [0, 0,0,0,0,0,0,0]
        one_hot[int(num)]=1
        all_one_hot.append(one_hot)
    return np.array(all_one_hot)


class ClasModle:
    def __init__(self):
        self.x=tf.placeholder(shape=[None,16,168,3],dtype=tf.float32,name='input')
        self.y_=tf.placeholder(shape=[None,8],dtype=tf.float32)
        self.dp=tf.placeholder(dtype=tf.float32)

        self.conv1_w = tf.Variable(tf.random_normal([3, 3, 3, 10], dtype=tf.float32, stddev=tf.sqrt(1 / 10)))
        self.conv1_b = tf.Variable(tf.zeros([10]))
        self.conv1_dw = tf.Variable(tf.random_normal([3, 3, 10, 1], dtype=tf.float32, stddev=tf.sqrt(1 / 10)))
        self.conv1_db = tf.Variable(tf.zeros([10]))

        self.conv2_w = tf.Variable(tf.random_normal([3, 3, 10, 16], dtype=tf.float32, stddev=tf.sqrt(1 / 32)))
        self.conv2_b = tf.Variable(tf.zeros([16]))
        self.conv2_dw = tf.Variable(tf.random_normal([1, 1, 16, 1], dtype=tf.float32, stddev=tf.sqrt(1 / 32)))

        # self.conv3_w=tf.Variable(tf.random_normal([3,3,16,32],dtype=tf.float32,stddev=tf.sqrt(1/16)))
        # self.conv3_b=tf.Variable(tf.zeros([32]))

        self.conv2_db = tf.Variable(tf.zeros([16]))
        #
        self.fc1_w = tf.Variable(tf.random_normal([1 * 11 * 16, 16], dtype=tf.float32, stddev=tf.sqrt(1 / 32)))
        self.fc1_b = tf.Variable(tf.zeros([16]))
        self.fc2_w = tf.Variable(tf.random_normal([16,  8], dtype=tf.float32, stddev=tf.sqrt(1 /  8)))
        self.fc2_b = tf.Variable(tf.zeros([8]))

    def forward(self):
        self.conv1 = tf.nn.relu(tf.layers.batch_normalization(
            tf.nn.conv2d(self.x, self.conv1_w, strides=[1, 2, 2, 1], padding='SAME') + self.conv1_b))  # 9,44,10
        # self.conv1_d = tf.nn.relu(tf.layers.batch_normalization(
        #     tf.nn.depthwise_conv2d(self.conv1, self.conv1_dw, strides=[1, 2, 2, 1], padding='SAME') + self.conv1_db
        # ))
        self.pool1 = tf.nn.max_pool(self.conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        self.conv2 = tf.nn.relu(tf.layers.batch_normalization(
            tf.nn.conv2d(self.pool1, self.conv2_w, strides=[1, 2, 2, 1], padding='SAME') + self.conv2_b))  # 2,7,16
        # self.conv2_d = tf.nn.relu(tf.layers.batch_normalization(
        #     tf.nn.depthwise_conv2d(self.conv2, self.conv2_dw, strides=[1, 2, 2, 1], padding='SAME') + self.conv2_db
        # ))
        self.pool2 = tf.nn.max_pool(self.conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # self.conv3=tf.nn.relu(tf.layers.batch_normalization(
        #     tf.nn.conv2d(self.conv2_d,self.conv3_w,strides=[1,2,2,1],padding='SAME') +self.conv3_b))
        # self.pool3=tf.nn.max_pool(self.conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        print(self.pool2)

        self.flat = tf.reshape(self.pool2, [-1, 1 * 11 * 16])

        self.fc1=tf.nn.relu(tf.layers.batch_normalization(tf.matmul(self.flat,self.fc1_w)+self.fc1_b))
        self.fc1_d=tf.nn.dropout(self.fc1,keep_prob=self.dp)
        self.output=tf.matmul(self.fc1,self.fc2_w)+self.fc2_b
        self.output=tf.reshape(self.output,[-1,8],name='output')

    def backward(self):
        self.loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output,labels=self.y_))
                                 # +0.00001 * tf.reduce_sum(tf.pow(self.conv1_w, 2))+0.00001 * tf.reduce_sum(tf.pow(self.conv2_w, 2))+
                                 # 0.00001 * tf.reduce_sum(tf.pow(self.fc1_w, 2)))
        # global_ = tf.Variable(tf.constant(0))
        # lr = tf.train.exponential_decay(0.001,30000, 300, 0.96, staircase=True)
        self.op=tf.train.AdamOptimizer(0.0002).minimize(self.loss)
        self.out_arg=tf.argmax(self.output,axis=1)
        self.label_arg=tf.argmax(self.y_,axis=1)

        self.acc=tf.reduce_mean(tf.cast(tf.equal(self.out_arg,self.label_arg),'float'))


if __name__=='__main__':
    net=ClasModle()
    net.forward()
    net.backward()
    init=tf.global_variables_initializer()

    train_data = train_shuffle_batch(train_filename, [16, 168, 3], 64)
    test_data = test_shuffle_batch(test_filename, [16, 168, 3], 20000)

    saver=tf.train.Saver()
    with  tf.Session() as sess:
        coord = tf.train.Coordinator()  # 创建一个协调器，管理线程
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        sess.run(init)
        x=[]
        y=[]
        saver.restore(sess,'modle_save/cls_modle1.cptk')
        count = 0
        plt.ion()
        for i in range(30000):
            train_img,train_label,train_name=sess.run(train_data)
            train_label=one_hott(train_label.tolist())
            train_img1=train_img/255-0.5

            input_x,conv1,conv2,pool1,pool2,_,train_loss,train_out,train_l,train_acc=sess.run([net.x ,net.conv1,net.conv2,net.pool1,net.pool2, net.op,net.loss,net.out_arg,net.label_arg, net.acc],
                                                              feed_dict={net.x:train_img1,net.y_:train_label,net.dp:0.5})



            # if i%100==0:
            #     plt.clf()
            #     plt.subplot(321)
            #     plt.imshow(input_x[0,:,:,0])
            #     plt.title(" {}th x ".format(i) )
            #
            #     plt.subplot(322)
            #     plt.imshow(conv1[0,:,:,0])
            #     plt.title(" {}th conv1 ".format(i))
            #     plt.subplot(323)
            #     plt.imshow(pool1[0, :, :, 0])
            #     plt.title(" {}th poo1 ".format(i))
            #     plt.subplot(324)
            #     plt.imshow(conv2[0, :, :, 0])
            #     plt.title(" {}th conv2 ".format(i))
            #     plt.subplot(325)
            #     plt.imshow(pool2[0, :, :, 0])
            #     plt.title(" {}th pool2 ".format(i))
            #     plt.pause(1)


            for ll in range(len(train_out)):
                if train_out[ll].tolist()==train_l[ll].tolist():
                    pass
                else:
                    # scipy.misc.imsave('/Users/wywy/Desktop/train_e/{}.jpg'.format(train_name[ll]),train_img[ll])
                    count+=1



            print('第{}次误差为 {} ，精度为 {} ，out: {} ,label: {}'.format(i,train_loss,train_acc,train_out[0],train_l[0]))

            if i%500==0:
            #     m_path='/Users/wywy/Desktop/手机扫描样例'
            #     for file in os.listdir(m_path):
            #         if file=='.DS_Store':
            #             os.remove(m_path+'/'+file)
            #         else:
            #             img1=cv.imread(m_path+'/'+file)
            #             img=img1/255-0.5
            #             img=img.reshape([-1,16,168,3])
            #
            #             m_out=sess.run(net.out_arg,feed_dict={net.x:img,net.dp:1.})
            #             scipy.misc.imsave('/Users/wywy/Desktop/save11/{}_{}.jpg'.format(file,m_out),img1)



                test_img, test_label, test_name = sess.run(test_data)
                test_label=one_hott(test_label.tolist())
                test_img1 = test_img / 255 - 0.5

                test_loss, test_out,test_l, test_acc = sess.run(
                    [net.loss, net.out_arg,net.label_arg, net.acc],
                    feed_dict={net.x: test_img1, net.y_: test_label,net.dp:1.})

                for ll1 in range(len(test_out)):
                    if test_out[ll1].tolist() == test_l[ll1].tolist():
                        pass
                    else:
                        # scipy.misc.imsave('/Users/wywy/Desktop/test_e/{}.jpg'.format(test_name[ll1]), test_img[ll1])
                        count+=1

                saver.save(sess,'modle_save/cls_modle1.cptk')
                #
                print('-------第{}次测试集误差为 {} ，精度为 {} ，out: {} ,label: {}--------'.format(i, test_loss, test_acc, test_out[0], test_l[0]))
                # graph_def = tf.get_default_graph().as_graph_def()
                # output_graph_def = tf.graph_util.convert_variables_to_constants(sess, graph_def,
                #                                                                 ['output'])
                #
                # with tf.gfile.GFile("./clas_modle.pb", 'wb') as f:
                #     f.write(output_graph_def.SerializeToString())



















