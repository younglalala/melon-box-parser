
from new_modle.clas_train_sample import *
from new_modle.clas_test_sample import *
from new_modle.one_hott import *

import matplotlib.pyplot as plt
import scipy.misc



class ClasModle:
    def __init__(self):
        self.x=tf.placeholder(shape=[None,16,168,1],dtype=tf.float32,name='input')
        self.y_=tf.placeholder(shape=[None,8],dtype=tf.float32)
        self.dp=tf.placeholder(dtype=tf.float32)

        self.conv1_w=tf.Variable(tf.random_normal([3,3,1,6],dtype=tf.float32,stddev=tf.sqrt(1/10)))
        self.conv1_b=tf.Variable(tf.zeros([6]))

        self.conv2_w=tf.Variable(tf.random_normal([3,3,6,10],dtype=tf.float32,stddev=tf.sqrt(1/16)))
        self.conv2_b=tf.Variable(tf.zeros([10]))

        self.conv3_w=tf.Variable(tf.random_normal([3,3,10,16],dtype=tf.float32,stddev=tf.sqrt(1/32)))
        self.conv3_b=tf.Variable(tf.zeros([16]))

        self.fc1_w=tf.Variable(tf.random_normal([1*11*16,32],dtype=tf.float32,stddev=tf.sqrt(1/128)))
        self.fc1_b=tf.Variable(tf.zeros([32]))

        self.fc2_w=tf.Variable(tf.random_normal([32,8],dtype=tf.float32,stddev=tf.sqrt(1/8)))
        self.fc2_b=tf.Variable(tf.zeros([8]))

    def forward(self):
        self.conv1=tf.nn.relu(tf.layers.batch_normalization( tf.nn.conv2d(self.x,self.conv1_w,strides=[1,1,1,1],padding='SAME')+self.conv1_b))
        self.conv1_d=tf.nn.dropout(self.conv1,keep_prob=self.dp)
        self.pool1=tf.nn.max_pool(self.conv1_d,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

        self.conv2=tf.nn.relu(tf.layers.batch_normalization(tf.nn.conv2d(self.pool1,self.conv2_w,strides=[1,2,2,1],padding='SAME')+self.conv2_b))
        self.conv2_d = tf.nn.dropout(self.conv2, keep_prob=self.dp)
        self.pool2=tf.nn.max_pool(self.conv2_d,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

        self.conv3=tf.nn.relu(tf.nn.conv2d(self.pool2,self.conv3_w,strides=[1,1,1,1],padding='SAME')+self.conv3_b)
        # self.conv3_d=tf.nn.dropout(self.conv3,keep_prob=self.dp)
        self.pool3=tf.nn.max_pool(self.conv3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
        print(self.pool3)
        self.flat=tf.reshape(self.pool3,[-1,1*11*16])

        self.fc1=tf.nn.relu(tf.layers.batch_normalization(tf.matmul(self.flat,self.fc1_w)+self.fc1_b))
        self.fc1_d=tf.nn.dropout(self.fc1,keep_prob=self.dp)
        self.output=tf.matmul(self.fc1,self.fc2_w)+self.fc2_b

    def backward(self):
        self.loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output,labels=self.y_))
                                 # +0.00001 * tf.reduce_sum(tf.pow(self.conv1_w, 2))+0.00001 * tf.reduce_sum(tf.pow(self.conv2_w, 2))+
                                 # 0.00001 * tf.reduce_sum(tf.pow(self.conv3_w, 2)))
        self.op=tf.train.AdamOptimizer(0.0002).minimize(self.loss)
        self.out_arg=tf.argmax(self.output,1)
        self.out_arg1=tf.reshape(self.out_arg,[-1],name='clas_outpot')

        self.label_arg=tf.argmax(self.y_,1)
        self.acc=tf.reduce_mean(tf.cast(tf.equal(self.out_arg,self.label_arg),'float'))


if __name__=='__main__':
    net=ClasModle()
    net.forward()
    net.backward()
    init=tf.global_variables_initializer()

    train_data = train_shuffle_batch(train_filename, [16, 168, 1], 256)
    test_data = test_shuffle_batch(test_filename, [16, 168, 1], 57300)

    saver=tf.train.Saver()
    with  tf.Session() as sess:
        coord = tf.train.Coordinator()  # 创建一个协调器，管理线程
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        sess.run(init)
        x=[]
        y=[]
        saver.restore(sess,'modle_save/cls_modle1.cptk')
        count = 0
        for i in range(10000):
            train_img,train_label,train_name=sess.run(train_data)
            train_label=train_label.reshape([-1]).tolist()
            train_label=one_hot(train_label)
            train_img1=train_img/255-0.5

            _,train_loss,train_out,train_l,train_acc=sess.run([net.op,net.loss,net.out_arg,net.label_arg,net.acc],
                                                              feed_dict={net.x:train_img1,net.y_:train_label,net.dp:0.6})

            train_flase='/Users/wywy/Desktop/train_flase'

            for train_index in range(len(train_out)):
                if train_out.tolist()[train_index]==train_l.tolist()[train_index]:
                    pass
                else:
                    image = np.expand_dims(train_img[train_index], axis=2)
                    image = np.concatenate((image, image, image), axis=2)
                    image=image.reshape([16,168,3])

                    # scipy.misc.imsave(train_flase+'/'+str(count)+'_'+str(train_l.tolist()[train_index])+'_'+str(train_out.tolist()[train_index])+'.jpg',image)
                    # count+=1




            # x.append(i)
            # y.append(train_loss)
            # plt.plot(x,y,'red')
            # plt.pause(0.1)
            # plt.clf()


            print('第{}次误差为 {} ，精度为 {} ，out: {} ,label: {}'.format(i,train_loss,train_acc,train_out[0],train_l[0]))

            if i%500==0:
                test_img, test_label, test_name = sess.run(test_data)
                test_label = test_label.reshape([-1]).tolist()
                test_label = one_hot(test_label)
                test_img1 = test_img / 255 - 0.5

                test_loss, test_out, test_l, test_acc = sess.run(
                    [ net.loss, net.out_arg, net.label_arg, net.acc],
                    feed_dict={net.x: test_img1, net.y_: test_label,net.dp:1.})
                saver.save(sess,'modle_save/cls_modle1.cptk')

                print('-------第{}次测试集误差为 {} ，精度为 {} ，out: {} ,label: {}--------'.format(i, test_loss, test_acc, test_out[0], test_l[0]))
                graph_def = tf.get_default_graph().as_graph_def()
                output_graph_def = tf.graph_util.convert_variables_to_constants(sess, graph_def,
                                                                                ['clas_outpot'])

                with tf.gfile.GFile("./clas_modle1.pb", 'wb') as f:
                    f.write(output_graph_def.SerializeToString())



















