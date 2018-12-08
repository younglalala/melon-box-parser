import os
import cv2
import numpy as np

detect_path='/Users/wywy/Desktop/detect_test'
class DetectSample:
    def __init__(self,detect_path):
        self.all_img=[]
        self.all_lable=[]
        for file in os.listdir(detect_path):
            if file=='.DS_Store':
                os.remove(detect_path+'/'+file)
            else:
                name=file.split('.')[0].split('_')[-1]
                img=cv2.imread(detect_path+'/'+file)/255-0.5
                self.all_img.append(img)
                self.all_lable.append(name)

    def get_batch(self,batch_size):
        self.batch_img=[]
        self.batch_lable=[]
        for i in range(batch_size):
            index=np.random.randint(len(self.all_lable))
            self.batch_img.append(self.all_img[index])
            self.batch_lable.append(self.all_lable[index])

        return np.array(self.batch_img),np.array(self.batch_lable)


if __name__=='__main__':
    detect_sample=DetectSample(detect_path)
    img,label=detect_sample.get_batch(100)
    print(img.shape)
    print(label.shape)


