import tensorflow as tf
from  cls_and_cnn.test2_modle.sample import *
import scipy.misc
import matplotlib.pyplot as plt

import  tempfile
import subprocess
tf.contrib.lite.tempfile=tempfile
tf.contrib.lite.subprocess=subprocess

tf.set_random_seed(1)




class ClsCnn:
    def __init__(self):
        self.x=tf.placeholder(tf.float32,[None,16,168,3],name='input')   #图片大小自定义，对图片进行归一化处理。
        self.y_=tf.placeholder(tf.float32,[None,7])

        self.conv1_w = tf.Variable(tf.random_normal([3, 3, 3, 10], dtype=tf.float32, stddev=tf.sqrt(1/10)))
        self.conv1_b = tf.Variable(tf.zeros([10]))
        self.conv1_dw=tf.Variable(tf.random_normal([3,3,10,1],dtype=tf.float32,stddev=tf.sqrt(1/10)))
        self.conv2_db=tf.Variable(tf.zeros([10]))

        self.conv2_w = tf.Variable(tf.random_normal([3, 3, 10, 16], dtype=tf.float32, stddev=tf.sqrt(1/16)))
        self.conv2_b = tf.Variable(tf.zeros([16]))

        self.fc1=tf.Variable(tf.random_normal([1*10*16,128],dtype=tf.float32,stddev=tf.sqrt(1/512)))
        self.b1=tf.Variable(tf.zeros([128]))
        self.fc2=tf.Variable(tf.random_normal([128,7],dtype=tf.float32,stddev=tf.sqrt(1/7)))
        self.b2 = tf.Variable(tf.zeros([7]))

    def forward(self):

        self.conv1=tf.nn.leaky_relu(tf.layers.batch_normalization(
            tf.nn.conv2d(self.x,self.conv1_w,strides=[1,2,2,1],padding='SAME')+self.conv1_b))   #9,44,10
        self.conv1_d=tf.nn.leaky_relu(tf.layers.batch_normalization(
            tf.nn.depthwise_conv2d(self.conv1,self.conv1_dw,strides=[1,2,2,1],padding='SAME')
        ))
        self.pool1=tf.nn.max_pool(self.conv1_d,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
        self.conv2 = tf.nn.leaky_relu(tf.layers.batch_normalization(
            tf.nn.conv2d(self.conv1_d, self.conv2_w, strides=[1, 2, 2, 1], padding='SAME') + self.conv2_b) )  #2,7,16
        self.pool2=tf.nn.max_pool(self.conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')

        flat = tf.reshape(self.pool2, [-1, 1 * 10 * 16])

        self.y1 = tf.nn.leaky_relu(tf.layers.batch_normalization(tf.matmul(flat, self.fc1) + self.b1))

        self.out1=tf.matmul(self.y1, self.fc2) + self.b2

        self.out=tf.reshape(self.out1,[-1,7],name='output')


        return self.out


    def backward(self):

        self.loss=tf.reduce_mean((self.y_-self.out)**2)

        self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.loss)


if __name__=='__main__':

    net = ClsCnn()
    net.forward()
    net.backward()

    init = tf.global_variables_initializer()
    x=[]
    y=[]
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        saver.restore(sess,'./modle_save/train1.dpk')
        flag = 0
        for ii in range(10000):
            batch_size=128
            train_data = TrainData(train_img, train_label,train_filename)
            trainimg, trainlabel,train_file = train_data.__getitem__(batch_size)

            _,loss,out,label1=sess.run([net.optimizer,net.loss,net.out,net.y_],
                                     feed_dict={net.x:trainimg,net.y_:trainlabel})

            all_index = []
            for i in range(out.shape[0]):
                index=np.where(out[i]>=0.5)[0]
                index1=np.where(out[i]<0.5)[0]
                out[i][index]=1
                out[i][index1]=0
            correct_prediction=[]


            # 训练集识别图片错误
            # flase_img_path='/Users/wywy/Desktop/flase_img/'
            count=0
            for num in range(batch_size):
                if out[num].tolist() ==label1[num].tolist():
                    correct_prediction.append(1)
                else:
                    correct_prediction.append(0)
                    out_name=np.where(out[num]>0.9)[0].tolist()
                    label_name=np.where (label1[num]>0.9)[0].tolist()
                    # scipy.misc.imsave(flase_img_path+str(count)+str(out_name)+train_file[num]+'.jpg',trainimg[num])
                    count+=1

            accuracy = np.mean(np.array(correct_prediction))

            x.append(ii)
            y.append(loss)
            plt.plot(x,y,'red')
            plt.pause(0.01)
            plt.clf()

            # print('第{}次的误差为{},jingdu:{}'.format(ii,loss,accuracy))
            #

            if ii%5==0:
                # frozen_graphdef = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def,
                #                                                                ['output'])  # 这里 ['output']是输出tensor的名字
                # tflite_model = tf.contrib.lite.toco_convert(frozen_graphdef, [net.x], [
                #     net.out])  # 这里[input], [out]这里分别是输入tensor或者输出tensor的集合,是变量实体不是名字
                # open("objective_modle.tflite", "wb").write(tflite_model)
                #扫描数据测试
                test_data = TestData(test_img, test_label,test_filename)
                testimg, testdata,test_file = test_data.__getitem__(batch_size)
                loss1,out1=sess.run([net.loss,net.out],feed_dict={net.x:testimg,net.y_:testdata})

                test_all_index = []
                for ii in range(out1.shape[0]):
                    index1 = np.where(out1[ii] >= 0.47)[0]
                    index11 = np.where(out1[ii] < 0.47)[0]
                    out1[ii][index1] = 1
                    out1[ii][index11] = 0

                test_correct_prediction = []
                # test_flase_img_path = '/Users/wywy/Desktop/test_false/'
                count1 = 0
                for num1 in range(batch_size):
                    if out1[num1].tolist() == testdata[num1].tolist():
                        test_correct_prediction.append(1)
                    else:
                        test_correct_prediction.append(0)
                        out_name1 = np.where(out1[num1] > 0.9)[0].tolist()
                        label_name1 = np.where(testdata[num1] > 0.9)[0].tolist()
                        # scipy.misc.imsave(test_flase_img_path + str(count1) + str(out_name1) + test_file[num1] + '.jpg',
                        #                   testimg[num1])
                        count1 += 1

                test_accuracy = np.mean(np.array(test_correct_prediction))
                #
                #
                print('第{}次测试集的误差是{},jingdu:{}'.format(ii,loss1,test_accuracy))




                saver.save(sess,'./modle_save/train1.dpk')
                # graph_def = tf.get_default_graph().as_graph_def()
                # output_graph_def = tf.graph_util.convert_variables_to_constants(sess, graph_def,
                #                                                                     ['output'])
                #
                # with tf.gfile.GFile("./train.pb", 'wb') as f:
                #     f.write(output_graph_def.SerializeToString())













