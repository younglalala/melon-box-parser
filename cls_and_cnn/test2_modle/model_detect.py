import os
import scipy.misc

import numpy as np
import tensorflow as tf
import cv2 as cv



#输入数据要求：1.图片形状（16，168，3）,2.客观题图片最多7个选项。
#输出数据格式：1.形状为【8，7】的numpy数组，找到最后一维中大于阈值的数的索引即可得到该图片的选项


#测试数据采样

class Test2Sample:
    def __init__(self,img_path):
        self.all_img=[]
        for file in os.listdir(img_path):
            if file=='.DS_Store':
                os.remove(img_path+'/'+file)
            else:
                img=cv.imread(img_path+'/'+file)/255-0.5
                self.all_img.append(img)
    def get_batch(self,batch_size):
        self.batch_img=[]
        for i in range(batch_size):
            index=np.random.randint(len(self.all_img))
            self.batch_img.append(self.all_img[index])
        return np.array(self.all_img)



#侦测网络
class ModelDetect:
    def __init__(self):
        pass
    def modle(self,image):
        with tf.Graph().as_default():
            output_graph_def = tf.GraphDef()
            output_graph_path = "./objective_modle2.pb"

            with open(output_graph_path, "rb") as f:
                output_graph_def.ParseFromString(f.read())
                tf.import_graph_def(output_graph_def, name="")
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())

                input_x = sess.graph.get_tensor_by_name("input:0")
                model_output = sess.graph.get_tensor_by_name("output:0")


                output = sess.run(model_output, feed_dict={input_x: image})
                return output


if __name__=='__main__':
    img_path='/Users/wywy/Desktop/手机扫描样例'
    save_path='/Users/wywy/Desktop/save11'
    data=Test2Sample(img_path)
    mobiel_img=data.get_batch(1)
    print(mobiel_img.shape)
    net=ModelDetect()
    out=net.modle(mobiel_img)
    print(out)
    out_arg=np.argmax(out,axis=2)
    count=0
    for i in range(len(mobiel_img)):
        name=np.where(out_arg[i]<7)[0]
        print(name)
        scipy.misc.imsave(save_path+'/'+str(count)+'_'+str(name)+'.jpg',mobiel_img[i])
        count+=1








