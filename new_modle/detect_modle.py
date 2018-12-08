import scipy.misc
import cv2
import os
from skimage import io,data

import tensorflow as tf
import numpy as np
from new_modle.cnn0_test_sample import *
import scipy.misc

detect_path='/Users/wywy/Desktop/客观题分类/test1_img'
# dict_info=dict(zip(list('ABCDEFGX'),[0,1,2,3,4,5,6,7]))
# class DetectSample:
#     def __init__(self):
#         pass
#
#     def all_data(self,detect_path):
#         self.all_img = []
#         self.all_label = []
#         all_filename = []
#         for i in os.listdir(detect_path):
#             if i == '.DS_Store':
#                 os.remove(detect_path + '/' + i)
#             else:
#                 all_filename.append(i)
#         random.shuffle(all_filename)
#         for ii in all_filename:
#             sample = ii.split('.')[0].split('_')
#             c = list('ABCDEFGX')
#             ll = []
#             for s in sample:
#                 if s in c:
#                     ll.append(s)
#
#             label_info = []
#             for i in range(8):
#                 label_info.append(0)
#             for chioce in ll:
#                 index = dic_info.get(chioce)
#                 label_info[index] = 1
#             labell=np.where(np.array(label_info)>0.9)[0]
#             self.all_label.append(labell)
#             img = Image.open (detect_path + '/' + ii).convert('L')
#             img=np.array(img)/255-0.5
#             if img.shape[0] == 16 and img.shape[1] == 168:
#                 img = img.reshape([16, 168, 1])
#                 self.all_img.append(img)
#         return self.all_img,self.all_label
#
#     def get_batch(self,all_img,all_label,batch_size):
#         self.batch_img=[]
#         self.batch_label=[]
#         for i in range(batch_size):
#             index=np.random.randint(len(all_img))
#             self.batch_img.append(all_img[index])
#             self.batch_label.append(all_label[index])
#
#         return np.array(self.batch_img),np.array(self.batch_label)









# def get_test_batch(test_filename,batch_size):
#     test_data = test_shuffle_batch(test_filename, [16, 168, 1], batch_size)
#     with tf.Session() as sess:
#         coord = tf.train.Coordinator()  # 创建一个协调器，管理线程
#         threads = tf.train.start_queue_runners(coord=coord, sess=sess)
#
#         test_img, test_label, test_file_name = sess.run(test_data)
#
#         return test_img,test_label,test_file_name

class Detect:
    def __init__(self):
        pass
    def clas_md(self,iamge):
        with tf.Graph().as_default():
            output_graph_def = tf.GraphDef()
            output_graph_path = "./clas_modle.pb"

            with open(output_graph_path, "rb") as f:
                output_graph_def.ParseFromString(f.read())
                tf.import_graph_def(output_graph_def, name="")
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())

                input_x = sess.graph.get_tensor_by_name("input:0")
                cls_output = sess.graph.get_tensor_by_name("clas_outpot:0")

                cls=sess.run(cls_output,feed_dict= {input_x:iamge})

                return cls,iamge

    def cnn_md(self,cls_image,cls):
        clas_info=["./cnn0_modle.pb","./cnn1_modle.pb","./cnn2_modle.pb","./cnn3_modle.pb",
                   "./cnn4_modle.pb","./cnn5_modle.pb","./cnn6_modle.pb","./cnn7_modle.pb"]
        with tf.Graph().as_default():
            output_graph_def = tf.GraphDef()
            # for cc in cls:
            output_graph_path = clas_info[int(3)]

            with open(output_graph_path, "rb") as f:
                output_graph_def.ParseFromString(f.read())
                tf.import_graph_def(output_graph_def, name="")
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())

                cls_input_x = sess.graph.get_tensor_by_name("input:0")
                cls_output = sess.graph.get_tensor_by_name("output:0")
                output=sess.run(cls_output,{cls_input_x:cls_image})
                return output






img=Image.open('/Users/wywy/Desktop/test_3/10_D_A_C.jpg').convert('L')
img=np.array(img).reshape((1,16,168,1))
detect=Detect()
out=detect.cnn_md(img,3)
print(out)




# if __name__=='__main__':
#     detect=Detect()
#     train_acc=[]
#     count=0
#     test_data = test_shuffle_batch(test_filename, [16, 168, 1], 1)
#     with tf.Session() as sess:
        # coord = tf.train.Coordinator()  # 创建一个协调器，管理线程
        # threads = tf.train.start_queue_runners(coord=coord, sess=sess)
        # for i in range(20000):
        #
        #     test_img, test_label, test_file_name = sess.run(test_data)
        #     batch_img=test_img/255-0.5
        #
        #     clas,img=detect.clas_md(batch_img)
        #     print(clas)
        #
        #     out=detect.cnn_md(img,clas)
        #     out_arg=np.argmax(out,axis=2)[0]
        #
        #     test_image = np.expand_dims(batch_img[0], axis=2)
        #     test_image = np.concatenate((test_image, test_image, test_image), axis=2)
        #     test_image = test_image.reshape([16, 168, 3])
        #     scipy.misc.imsave('/Users/wywy/Desktop/save11/'+str(count)+'_'+str(out_arg)+'.jpg',test_image)
        #     count+=1


        #
        #     label=test_label
        #     label_arg=np.where(test_label[0]>0.9)[0]
        #     if len(label_arg.tolist())==0:
        #         label_arg=np.array([7])
        #     print(label_arg)
        #     print(out_arg)
        #
        #     print('*********{}*********'.format(i))
        #
        #     if label_arg.tolist()==out_arg.tolist():
        #         train_acc.append(1)
        #     else:
        #         train_acc.append(0)
        # print(np.mean(np.array(train_acc)))

    # test_image = np.expand_dims(batch_img[0], axis=2)
    # test_image = np.concatenate((test_image, test_image, test_image), axis=2)
    # test_image = test_image.reshape([16, 168, 3])
    # scipy.misc.imsave('/Users/wywy/Desktop/save11/1.jpg',test_image)










