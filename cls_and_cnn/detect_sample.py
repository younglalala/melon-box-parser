import cv2
import os

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

if __name__=='__main__':
    detect_sample=DetectSample(detect_path)
    batch_img,batch_label=detect_sample.get_batch(1000)
    # print(batch_label)
    # print(batch_img.shape)
