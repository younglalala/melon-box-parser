import os
import numpy
import scipy.misc
from PIL import Image
import numpy as np
import tensorflow as tf

img_path='/Users/wywy/Desktop/mobile_test'
def mobile_data(img_path):
    all_img=[]
    for file in os.listdir(img_path):
        if file=='.DS_Store':
            os.remove(img_path+'/'+file)
        else:
            img=Image.open(img_path+'/'+file).convert('L')
            img=np.array(img)
            img=img.reshape([16,168,1])
            all_img.append(img)
    return np.array(all_img)



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
            for cc in cls:
                output_graph_path = clas_info[int(cc)]

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
    detect = Detect()
    batch_data=mobile_data(img_path)
    count=0
    for img in batch_data:
        img=img.reshape([-1,16,168,1])
        clas,image=detect.clas_md(img)

        out=detect.cnn_md(image,[1])
        print(out)

        out_arg = np.argmax(out, axis=2)[0]

        test_image = np.expand_dims(image[0], axis=2)
        test_image = np.concatenate((test_image, test_image, test_image), axis=2)
        test_image = test_image.reshape([16, 168, 3])
        scipy.misc.imsave('/Users/wywy/Desktop/save11/' + str(count) + '_' + str(out_arg) + '.jpg', test_image)
        count += 1





