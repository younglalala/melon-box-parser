from new_modle.cnn0_test_sample import *
from new_modle.cnn0_train_sample import *

from new_modle.one_hotte import *

import matplotlib.pyplot as plt

import scipy.misc


class ClasModle0:
    def __init__(self):
        self.x=tf.placeholder(shape=[None,16,168,1],dtype=tf.float32,name='input')
        self.y_=tf.placeholder(shape=[None,1,8],dtype=tf.float32)

        self.conv1_w=tf.Variable(tf.random_normal([3,3,1,6],dtype=tf.float32,stddev=tf.sqrt(1/10)))
        self.conv1_b=tf.Variable(tf.zeros([6]))

        self.conv2_w=tf.Variable(tf.random_normal([3,3,6,10],dtype=tf.float32,stddev=tf.sqrt(1/16)))
        self.conv2_b=tf.Variable(tf.zeros([10]))

        # self.conv3_w=tf.Variable(tf.random_normal([3,3,16,32],dtype=tf.float32,stddev=tf.sqrt(1/32)))
        # self.conv3_b=tf.Variable(tf.zeros([32]))

        self.fc1_w=tf.Variable(tf.random_normal([1*11*10,64],dtype=tf.float32,stddev=tf.sqrt(1/64)))
        self.fc1_b=tf.Variable(tf.zeros([64]))

        self.fc2_w=tf.Variable(tf.random_normal([64,1*8],dtype=tf.float32,stddev=tf.sqrt(1/8*7)))
        self.fc2_b=tf.Variable(tf.zeros([1*8]))


    def forward(self):
        self.conv1=tf.nn.relu(tf.nn.conv2d(self.x,self.conv1_w,strides=[1,2,2,1],padding='SAME')+self.conv1_b)
        self.pool1=tf.nn.max_pool(self.conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
        #
        self.conv2=tf.nn.relu(tf.nn.conv2d(self.pool1,self.conv2_w,strides=[1,2,2,1],padding='SAME')+self.conv2_b)
        self.pool2=tf.nn.max_pool(self.conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
        #
        # self.conv3=tf.nn.relu(tf.nn.conv2d(self.pool2,self.conv3_w,strides=[1,1,1,1],padding='SAME')+self.conv3_b)
        # self.pool3=tf.nn.max_pool(self.conv3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
        print(self.pool2)
        self.flat=tf.reshape(self.pool2,[-1,1*11*10])

        self.fc1=tf.nn.relu(tf.matmul(self.flat,self.fc1_w)+self.fc1_b)
        self.output=tf.matmul(self.fc1,self.fc2_w)+self.fc2_b
        self.output=tf.reshape(self.output,[-1,1,8],name='output')

    def backward(self):
        self.loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output,labels=self.y_))
        self.op=tf.train.AdamOptimizer(0.001).minimize(self.loss)
        self.out_arg=tf.argmax(self.output,2)
        self.label_arg=tf.argmax(self.y_,2)
        self.acc=tf.reduce_mean(tf.cast(tf.equal(self.out_arg,self.label_arg),'float'))


if __name__=='__main__':
    net=ClasModle0()
    net.forward()
    net.backward()
    init=tf.global_variables_initializer()

    train_data = train_shuffle_batch(train_filename, [16, 168, 1], 256)
    test_data = test_shuffle_batch(test_filename, [16, 168, 1], 6000)

    saver=tf.train.Saver()
    with  tf.Session() as sess:
        coord = tf.train.Coordinator()  # 创建一个协调器，管理线程
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        sess.run(init)
        x=[]
        y=[]
        # saver.restore(sess,'modle_save/cnn7_modle.cptk')
        count = 0
        for i in range(10000):
            train_img, train_label, train_file_name = sess.run(train_data)
            print(train_label[0])



            train_img1 = train_img / 255 - 0.5
            all_train_label = []
            for kk in range(len(train_img)):
                index = np.where(train_label[kk] > 0.9)[0]
                train_l = onehott(index.tolist(),0)
                all_train_label.append(train_l)

            # print(len(np.array(all_train_label).shape))
            # if len(np.array(all_train_label).shape)!=4:
            #     # print(np.array(all_train_label))
            #     for bb in range(len(all_train_label)):
            #         if len(all_train_label[bb])==0:
            #             print(train_file_name[bb])


            _, loss, out, label1, ac, out_arg,label_arg = sess.run(
                [net.op, net.loss, net.output, net.y_, net.acc, net.out_arg,net.label_arg],
                feed_dict={net.x: train_img1, net.y_:np.array(all_train_label) })



            tr_acc = []
            for inn1 in range(len(out_arg)):
                if out_arg.tolist()[inn1] == label_arg.tolist()[inn1]:
                    tr_acc.append(1)
                else:
                    image = np.expand_dims(train_img[inn1], axis=2)
                    image = np.concatenate((image, image, image), axis=2)
                    image = image.reshape([16, 168, 3])
                    # print('/Users/wywy/Desktop/train_e' + '/' + bytes.decode(train_file_name[inn1]) + '_' + str(
                    #     out_arg.tolist()[inn1]) + '.jpg')
                    # scipy.misc.imsave(
                    #     '/Users/wywy/Desktop/train_e' + '/' + bytes.decode(train_file_name[inn1]) + '_' + str(
                    #         out_arg.tolist()[inn1]) + '.jpg', image)

                    tr_acc.append(0)
            train_acc = np.mean(np.array(tr_acc))

            # x.append(i)
            # y.append(loss)
            # plt.plot(x,y,'red')
            # plt.pause(0.1)
            # plt.clf()

            #

            print('第{}次误差为{}，精度：{}'.format(i, loss, train_acc))
            # print(np.where(out_arg[0] < 7)[0], '******', np.where(train_label[0] > 0.9)[0])
            print(out_arg[0],label_arg[0])
            if i % 300 == 0:

                test_img1, test_label, test_train_file = sess.run(test_data)
                test_img = test_img1 / 255 - 0.5
                all_test_label = []
                for kkk in range(len(test_img)):
                    index1 = np.where(test_label[kkk] > 0.9)[0]
                    test_l = onehott(index1.tolist(),0)
                    # print(test_l)
                    all_test_label.append(test_l)

                test_loss, test_arg, test_label_arg,test_acc = sess.run([net.loss, net.out_arg,net.label_arg, net.acc],
                                                         feed_dict={net.x: test_img, net.y_: np.array(all_test_label)})
                t_acc = []
                m_save = '/Users/wywy/Desktop/test_e'
                for inn in range(len(test_arg)):

                    if test_arg.tolist()[inn] == test_label_arg.tolist()[inn]:
                        t_acc.append(1)
                    else:
                        t_acc.append(0)
                        test_image = np.expand_dims(test_img[inn], axis=2)
                        test_image = np.concatenate((test_image, test_image, test_image), axis=2)
                        test_image = test_image.reshape([16, 168, 3])
                        # scipy.misc.imsave(m_save + '/' + str(test_arg.tolist()[inn]) + '_' + bytes.decode(test_train_file[inn]),
                        #                   test_image)

                test_acc = np.mean(np.array(t_acc))
                print('测试集准确度', test_acc, '--------------------——————————————————')
                # saver.save(sess,'modle_save/cnn0_modle.cptk')
                # graph_def = tf.get_default_graph().as_graph_def()
                # output_graph_def = tf.graph_util.convert_variables_to_constants(sess, graph_def,
                #                                                                 ['output'])
                #
                # with tf.gfile.GFile("./cnn0_modle.pb", 'wb') as f:
                #     f.write(output_graph_def.SerializeToString())
