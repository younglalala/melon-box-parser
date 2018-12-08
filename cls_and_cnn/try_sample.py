import os
import numpy as np
import matplotlib.image as iming

true_lable_path='/Users/wywy/Desktop/cnn_test2'
class Sample:
    def __init__(self,path):
        self.path=path
        self.all_images=[]
        self.all_lable=[]
        for file in os .listdir(true_lable_path):
            name=file.split('.')[0].split('_')[-1]
            self.all_lable.append(name)
            img=iming.imread(true_lable_path+'/'+file)/255.0-0.5
            self.all_images.append(img)
        self.len=len(self.all_images)
    def shuffe_batch(self,batch_size):
        batch_img=[]
        batch_lable=[]
        for indx in range(batch_size):
            self.dix=np.random.randint(self.len)
            batch_img.append(self.all_images[self.dix])
            batch_lable.append(self.all_lable[self.dix])
        return batch_img,batch_lable
sample=Sample(true_lable_path)
if __name__=='__main__':
    img,lable=sample.shuffe_batch(100)

