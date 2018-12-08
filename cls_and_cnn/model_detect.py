import scipy.misc
import cv2
import os

import tensorflow as tf
import numpy as np

detect_path='/Users/wywy/Desktop/detect_test'
class DetectSample:

    def __init__(self,detect_path):
        self.all_img = []
        self.all_label = []

        for file in os.listdir(detect_path):
            if file=='.DS_Store':
                os.remove(detect_path+'/'+file)
            else:
                lable=file.split('.')[0].split('_')[-1]
                if lable=='X'or lable=='0':
                    lable1=0
                else:
                    lable1=1
                img=cv2.imread(detect_path+'/'+file)/255-0.5
                self.all_img.append(img)
                self.all_label.append(lable1)

    def get_batch(self,batch_size):
        self.batch_img=[]
        self.batch_label=[]
        for i in range(batch_size):
            index=np.random.randint(len(self.all_label))
            self.batch_img.append(self.all_img[index])
            self.batch_label.append(self.all_label[index])

        return np.array(self.batch_img),np.array(self.batch_label)


class Detect:
    def __init__(self):
        pass
    def clas_md(self,iamge,clsvalue):
        with tf.Graph().as_default():
            output_graph_def = tf.GraphDef()
            output_graph_path = "./cls_modle.pb"

            with open(output_graph_path, "rb") as f:
                output_graph_def.ParseFromString(f.read())
                tf.import_graph_def(output_graph_def, name="")
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())

                input_x = sess.graph.get_tensor_by_name("cls_input:0")
                cls_output = sess.graph.get_tensor_by_name("cls_output:0")

                cls=sess.run(cls_output,feed_dict= {input_x:iamge})
                cls_bool = np.greater_equal(cls, clsvalue)

                bad_img=np.less(cls,clsvalue)
                bad_idx=np.where(bad_img)
                list_bad_idx=bad_idx[0].tolist()
                badd_img = (iamge[list_bad_idx])

                cls_ids = np.where(cls_bool)
                list_idx = cls_ids[0].tolist()
                good_img = (iamge[list_idx])
                return good_img,badd_img
    def cnn_md(self,cls_image):
        with tf.Graph().as_default():
            output_graph_def = tf.GraphDef()
            output_graph_path = "./cnn_modle.pb"

            with open(output_graph_path, "rb") as f:
                output_graph_def.ParseFromString(f.read())
                tf.import_graph_def(output_graph_def, name="")
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())

                cls_input_x = sess.graph.get_tensor_by_name("input:0")
                cls_output = sess.graph.get_tensor_by_name("output:0")
                output=sess.run(cls_output,{cls_input_x:cls_image})
                return output


if __name__=='__main__':
    detect=Detect()
    detect_sample = DetectSample(detect_path)
    batch_img, batch_label = detect_sample.get_batch(1000)
    for ii in range(len(batch_label)):
        if batch_label[ii]==0:
            scipy.misc.imsave('/Users/wywy/Desktop/label0/{}_0.jpg'.format(ii), batch_img[ii])

    good,bad=detect.clas_md(batch_img,0.26)    #分类出来好的数据和不好的数据
    out=detect.cnn_md(good)    #输出分类的ABCD选项
    for k in range(len(good)):
        if out[k]==0:
            scipy.misc.imsave('/Users/wywy/Desktop/test1/{}_A.jpg'.format(k), good[k])
        elif out[k]==1:
            scipy.misc.imsave('/Users/wywy/Desktop/test1/{}_B.jpg'.format(k), good[k])
        elif out[k] == 2:
            scipy.misc.imsave('/Users/wywy/Desktop/test1/{}_C.jpg'.format(k), good[k])
        elif out[k]==3:
            scipy.misc.imsave('/Users/wywy/Desktop/test1/{}_D.jpg'.format(k), good[k])
        elif out[k]==4:
            scipy.misc.imsave('/Users/wywy/Desktop/test1/{}_X.jpg'.format(k), good[k])
    for jj in range(len(bad)):
        scipy.misc.imsave('/Users/wywy/Desktop/out0/{}_0.jpg'.format(jj), bad[jj])




