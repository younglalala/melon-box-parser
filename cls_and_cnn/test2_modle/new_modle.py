import tensorflow as tf
from  cls_and_cnn.test2_modle.train_sample import *
from  cls_and_cnn.test2_modle.test_sample import *
from cls_and_cnn.test2_modle.one_hotte import *
import scipy.misc
import matplotlib.pyplot as plt
import cv2 as cv
import math

import  tempfile
import subprocess
tf.contrib.lite.tempfile=tempfile
tf.contrib.lite.subprocess=subprocess

tf.set_random_seed(1)


# mobeil_test='/Users/wywy/Desktop/cut'
mobeil_test='/Users/wywy/Desktop/mm'
def mobiel_test_img(mobeil_test):
    all_img = []
    for file in os.listdir(mobeil_test):

        if file=='.DS_Store':
            os.remove(mobeil_test+'/'+file)
        else:
            img=cv.imread(mobeil_test+'/'+file)
            all_img.append(img)
    return np.array(all_img)

batch_size=128
class ClsCnn:
    def __init__(self):
        self.x = tf.placeholder(tf.float32, [None, 16, 168, 3], name='input')  # 图片大小自定义，对图片进行归一化处理。
        self.y_ = tf.placeholder(tf.float32, [None, 7,8])
        # self.dp=tf.placeholder(tf.float32)

        self.conv1_w = tf.Variable(tf.random_normal([3, 3, 3, 10], dtype=tf.float32, stddev=tf.sqrt(1 / 16)))
        self.conv1_b = tf.Variable(tf.zeros([10]))
        self.conv1_dw = tf.Variable(tf.random_normal([3, 3, 10, 1], dtype=tf.float32, stddev=tf.sqrt(1 / 10)))
        self.conv1_db = tf.Variable(tf.zeros([10]))

        self.conv2_w = tf.Variable(tf.random_normal([3, 3, 10, 32], dtype=tf.float32, stddev=tf.sqrt(1 / 16)))
        self.conv2_b = tf.Variable(tf.zeros([32]))
        self.conv2_dw = tf.Variable(tf.random_normal([1, 1, 32, 1], dtype=tf.float32, stddev=tf.sqrt(1 / 16)))

        # self.conv3_w=tf.Variable(tf.random_normal([3,3,16,32],dtype=tf.float32,stddev=tf.sqrt(1/16)))
        # self.conv3_b=tf.Variable(tf.zeros([32]))

        self.conv2_db = tf.Variable(tf.zeros([32]))
        #
        self.fc1 = tf.Variable(tf.random_normal([1 * 6 * 32, 64], dtype=tf.float32, stddev=tf.sqrt(1 / 16)))
        self.b1 = tf.Variable(tf.zeros([64]))
        self.fc2 = tf.Variable(tf.random_normal([64, 7*8], dtype=tf.float32, stddev=tf.sqrt(1 / 7*8)))
        self.b2 = tf.Variable(tf.zeros([7*8]))

    def forward(self):
        self.conv1 = tf.nn.relu(tf.layers.batch_normalization(
            tf.nn.conv2d(self.x, self.conv1_w, strides=[1, 2, 2, 1], padding='SAME') + self.conv1_b))  # 9,44,10
        self.conv1_d = tf.nn.relu(tf.layers.batch_normalization(
            tf.nn.depthwise_conv2d(self.conv1, self.conv1_dw, strides=[1, 2, 2, 1], padding='SAME')+self.conv1_db
        ))
        self.pool1 = tf.nn.max_pool(self.conv1_d, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        self.conv2 = tf.nn.relu(tf.layers.batch_normalization(
            tf.nn.conv2d(self.pool1, self.conv2_w, strides=[1, 1, 1, 1], padding='SAME') + self.conv2_b))  # 2,7,16
        self.conv2_d = tf.nn.relu(tf.layers.batch_normalization(
            tf.nn.depthwise_conv2d(self.conv2, self.conv2_dw, strides=[1, 2, 2, 1], padding='SAME') + self.conv2_db
        ))
        self.pool2 = tf.nn.max_pool(self.conv2_d, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # self.conv3=tf.nn.relu(tf.layers.batch_normalization(
        #     tf.nn.conv2d(self.conv2_d,self.conv3_w,strides=[1,2,2,1],padding='SAME') +self.conv3_b))
        # self.pool3=tf.nn.max_pool(self.conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        print(self.pool2)

        flat = tf.reshape(self.pool2, [-1, 1 * 6 * 32])

        self.y1 = tf.nn.relu(tf.layers.batch_normalization(tf.matmul(flat, self.fc1) + self.b1))
        # self.y1=tf.nn.dropout(self.y1,keep_prob=self.dp)

        self.out1 = tf.matmul(self.y1, self.fc2) + self.b2

        self.out = tf.reshape(self.out1, [-1, 7,8], name='output')

        return self.out


    def backward(self):

        self.loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_,logits=self.out))
        # self.loss = tf.reduce_mean(tf.pow(self.out - self.y_, 2) + 0.00001 * tf.reduce_sum(tf.pow(self.conv1_w, 2))
        #                            + 0.00001 * tf.reduce_sum(tf.pow(self.w1, 2))
        #                            + 0.00001 * tf.reduce_sum(tf.pow(self.w3, 2)))
        self.correct_prediction = tf.equal(tf.argmax(self.out, 2), tf.argmax(self.y_, 2))
        self.out_argmax = tf.argmax(self.out, 2)
        self.optimizer = tf.train.AdamOptimizer(0.0006).minimize(self.loss)
        self.rst = tf.cast(self.correct_prediction, "float")
        self.accuracy = tf.reduce_mean(self.rst)


if __name__=='__main__':

    net = ClsCnn()
    net.forward()
    net.backward()

    init = tf.global_variables_initializer()
    x=[]
    y=[]
    saver = tf.train.Saver()
    train_data=train_shuffle_batch(train_filename,[16, 168, 3], 128)
    test_data = test_shuffle_batch(test_filename, [16, 168, 3], 23000)
    with tf.Session() as sess:
        coord = tf.train.Coordinator()  # 创建一个协调器，管理线程
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)
        sess.run(init)
        saver.restore(sess,'./modle_save/m_train2.dpk')
        # # flag = 0
        for ii in range(50000):
            train_img,train_label,train_file_name=sess.run(train_data)

            train_img1=train_img/255-0.5
            all_train_label=[]
            for kk in range(len(train_img)):
                index=np.where(train_label[kk]>0.9)[0]
                train_l=onehott(index.tolist())
                all_train_label.append(train_l)



            _,loss,out,label1,ac,out_arg=sess.run([net.optimizer,net.loss,net.out,net.y_,net.accuracy,net.out_argmax],
                                     feed_dict={net.x:train_img1,net.y_:np.array(all_train_label)})

            tr_acc = []
            for inn1 in range(len(out_arg)):
                train_out = np.where(out_arg[inn1] < 7)[0]

                train_ll = np.where(train_label[inn1] > 0.9)[0]
                if train_out.tolist() == train_ll.tolist():
                    tr_acc.append(1)
                else:
                    # print('/Users/wywy/Desktop/train_e' + '/' + bytes.decode(train_file_name[inn1]) + '_' + str(train_out.tolist()) + '.jpg')
                    # scipy.misc.imsave('/Users/wywy/Desktop/train_e'+'/'+bytes.decode(train_file_name[inn1])+'_'+str(train_out.tolist())+'.jpg',train_img[inn1])


                    tr_acc.append(0)
            train_acc = np.mean(np.array(tr_acc))
            #

            print('{} epoch  第{}次误差为{}，精度：{}'.format(int(math.ceil(ii*128/500000)),ii,loss,train_acc))
            print(np.where(out_arg[0] < 7)[0],'******',np.where(train_label[0]>0.9)[0])
            if ii%500==0:
                test_img1, test_label,test_train_file = sess.run(test_data)
                test_img = test_img1 / 255 - 0.5
                all_test_label = []
                for kkk in range(len(test_img)):
                    index1 = np.where(test_label[kkk] > 0.9)[0]
                    test_l = onehott(index1.tolist())
                    all_test_label.append(test_l)

                test_loss,test_arg,test_acc=sess.run([net.loss,net.out_argmax,net.accuracy],
                                                     feed_dict={net.x:test_img,net.y_:np.array(all_test_label)})
                t_acc=[]
                m_save = '/Users/wywy/Desktop/test_e'
                for inn in range(len(test_arg)):
                    test_out=np.where(test_arg[inn]<7)[0]
                    test_ll=np.where(test_label[inn] > 0.9)[0]
                    if test_out.tolist()==test_ll.tolist():
                        t_acc.append(1)
                    else:
                        t_acc.append(0)
                        # scipy.misc.imsave(m_save+'/'+str(test_out)+'_'+bytes.decode(test_train_file[inn]),test_img1[inn])


                test_acc=np.mean(np.array(t_acc))
                print('测试集误差{} 测试集准确度{}'.format(test_loss,test_acc),'--------------------——————————————————')
                saver.save(sess,'./modle_save/m_train2.dpk')

                # m_save='/Users/wywy/Desktop/手机扫描样例'
                # mobile_test11=mobiel_test_img(mobeil_test)
                # mobile_test1=mobile_test11/255-0.5
                # m_out,oo=sess.run([net.out_argmax,net.out],feed_dict={net.x:mobile_test1})
                # print(m_out)
                # print(oo)
                # c=0
                # for mm in range(len(m_out)):
                #     out_la=np.where(m_out[mm]<7)[0]
                #     scipy.misc.imsave(m_save +'/'+ str(c) + str(out_la)+ '.jpg',
                #                       mobile_test1[mm])
                #     c+=1


                # saver.save(sess,'./modle_save/train6.dpk')
                # frozen_graphdef = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def,
                #                                                                ['output'])  # 这里 ['output']是输出tensor的名字
                # tflite_model = tf.contrib.lite.toco_convert(frozen_graphdef, [net.x], [
                #     net.out])  # 这里[input], [out]这里分别是输入tensor或者输出tensor的集合,是变量实体不是名字
                # open("objective_modle3.tflite", "wb").write(tflite_model)

                #
                graph_def = tf.get_default_graph().as_graph_def()
                output_graph_def = tf.graph_util.convert_variables_to_constants(sess, graph_def,
                                                                                ['output'])

                with tf.gfile.GFile("./objective_modle2.pb", 'wb') as f:
                    f.write(output_graph_def.SerializeToString())






















            #
            # all_index = []
            # for i in range(out.shape[0]):
            #     index=np.where(out[i]>=0.48)[0]
            #     index1=np.where(out[i]<0.48)[0]
            #     out[i][index]=1
            #     out[i][index1]=0
            # correct_prediction=[]
            #
            #
            # #训练集识别图片错误
            # # flase_img_path='/Users/wywy/Desktop/flase_img/'
            # count=0
            # for num in range(len(train_img)):
            #     if out[num].tolist() ==label1[num].tolist():
            #         correct_prediction.append(1)
            #     else:
            #         correct_prediction.append(0)
            #         out_name=np.where(out[num]>0.9)[0].tolist()
            #         label_name=np.where (label1[num]>0.9)[0].tolist()
            #         # scipy.misc.imsave(flase_img_path+str(count)+str(out_name)+train_file[num]+'.jpg',trainimg[num])
            #         count+=1
            #
            # accuracy = np.mean(np.array(correct_prediction))

            # x.append(ii)
            # y.append(loss)
            # plt.plot(x,y,'red')
            # plt.pause(0.01)
            # plt.clf()
            # print(loss)


            # print('第{}次的误差为{},jingdu:{}'.format(ii,loss,accuracy))
            #

            # if ii%50==0:
                #生成tf.lite 文件
                # frozen_graphdef = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def,
                #                                                                ['output'])  # 这里 ['output']是输出tensor的名字
                # tflite_model = tf.contrib.lite.toco_convert(frozen_graphdef, [net.x], [
                #     net.out])  # 这里[input], [out]这里分别是输入tensor或者输出tensor的集合,是变量实体不是名字
                # open("objective_modle2.tflite", "wb").write(tflite_model)
                # saver.save(sess,'./modle_save/train3.dpk')
                #扫描数据测试
                # test_img, test_label = sess.run(test_data)
                # test_img = test_img/ 255 - 0.5
                # loss1,out1=sess.run([net.loss,net.out],feed_dict={net.x:test_img,net.y_:test_label})
                #
                # test_all_index = []
                # for ii in range(out1.shape[0]):
                #     index1 = np.where(out1[ii] >= 0.47)[0]
                #     index11 = np.where(out1[ii] < 0.47)[0]
                #     out1[ii][index1] = 1
                #     out1[ii][index11] = 0
                #
                # test_correct_prediction = []
                # # test_flase_img_path = '/Users/wywy/Desktop/test_false/'
                # count1 = 0
                # for num1 in range(batch_size):
                #     if out1[num1].tolist() == test_label[num1].tolist():
                #         test_correct_prediction.append(1)
                #     else:
                #         test_correct_prediction.append(0)
                #         out_name1 = np.where(out1[num1] > 0.9)[0].tolist()
                #         label_name1 = np.where(test_label[num1] > 0.9)[0].tolist()
                #         # scipy.misc.imsave(test_flase_img_path + str(count1) + str(out_name1) + test_file[num1] + '.jpg',
                #         #                   testimg[num1])
                #         count1 += 1
                #
                # test_accuracy = np.mean(np.array(test_correct_prediction))
                # #
                #
                # print('第{}次测试集的误差是{},jingdu:{}'.format(ii,loss1,test_accuracy))

                # mobile_test11=mobiel_test_img(mobeil_test)
                # mobile_test1=mobile_test11/255-0.5
                # m_out=sess.run(net.out,feed_dict={net.x:mobile_test1})
                # print(m_out)
                # for ii in range(m_out.shape[0]):
                #     index1 = np.where(m_out[ii] >= 0.47)[0]
                #     # index11 = np.where(m_out[ii] < 0.47)[0]
                #     # m_out[ii][index1] = 1
                #     # m_out[ii][index1]=0
                #     scipy.misc.imsave(mobeil_test + str(ii) + str(index1)+ '.jpg',
                #                       mobile_test11[ii])







                # saver.save(sess,'./modle_save/train4.dpk')
                # graph_def = tf.get_default_graph().as_graph_def()
                # output_graph_def = tf.graph_util.convert_variables_to_constants(sess, graph_def,
                #                                                                     ['output'])
                #
                # with tf.gfile.GFile("./train.pb", 'wb') as f:
                #     f.write(output_graph_def.SerializeToString())













