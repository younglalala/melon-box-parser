import os
import cv2 as cv
import numpy as np

from torch.utils.data import  Dataset



test_path='/Users/wywy/Desktop/test1_img'
train_path='/Users/wywy/Desktop/训练完成模型/客观题数据（单、多选）/train1_img'
# test2_path='/Users/wywy/Desktop/test2_img'


def file_info(img_path):
    option=['A','B','C','D','E','F','G','X']
    index=[]
    for i in range(len(option)):
        index.append(i)
    dic_info = dict(zip(option, index))

    return dic_info
dic_info=file_info(train_path)
print(dic_info)

def get_info(dic_info,img_path):
    all_img=[]
    all_label=[]
    all_filename=[]
    for file in os.listdir(img_path):
        if file=='.DS_Store':
            os.remove(img_path+'/'+file)
        else:
            name = file.split('.')[0].split('_')[1:]
            img=cv.imread(img_path+'/'+file)/255-0.5

            label_info = []
            for i in range(8):
                label_info.append(0)
            for chioce in name:
                index = dic_info.get(chioce)
                label_info[index]=1
            label_info.pop()
            print(label_info)
            all_label.append(label_info)
            all_img.append(img)
            all_filename.append(file)

    return all_img,all_label,all_filename
# # #
test_img,test_label,test_filename=get_info(dic_info,test_path)

train_img,train_label,train_filename=get_info(dic_info,train_path)



class TestData(Dataset):
    def __init__(self,all_img,all_label,all_filename):
        self.all_img=all_img
        self.all_label=all_label
        self.all_filename=all_filename
    def __len__(self):
        return len(self.all_img)
    def __getitem__(self, batch_size):
        batch_img=[]
        batch_label=[]
        batch_filename=[]
        for i in range(batch_size):
            index=np.random.randint(len(self.all_img))
            batch_img.append(self.all_img[index])
            batch_label.append(self.all_label[index])
            batch_filename.append(self.all_filename[index])
        return np.array(batch_img) ,np.array(batch_label),batch_filename

class TrainData(Dataset):
    def __init__(self, all_img, all_label,all_filename):
        self.all_img = all_img
        self.all_label = all_label
        self.all_filename=all_filename

    def __len__(self):
        return len(self.all_img)

    def __getitem__(self, batch_size):
        batch_img = []
        batch_label = []
        batch_filename=[]
        for i in range(batch_size):
            index = np.random.randint(len(self.all_img))
            batch_img.append(self.all_img[index])
            batch_label.append(self.all_label[index])
            batch_filename.append(self.all_filename[index])
        return np.array(batch_img), np.array(batch_label),batch_filename




# test2_sample=Test2Sample(test2_path)


if __name__=='__main__':
    pass
    #
    # test_data=TestData(test_img,test_label)
    # testimg,testlabel=test_data.__getitem__(100)
    # print(testimg.shape)
    # print(testlabel.shape)
    #
    # train_data=TrainData(train_img,train_label)
    # trainimg,trainlabel=train_data.__getitem__(10)
    # print(trainimg.shape,trainlabel.shape)
    # print(trainlabel)










    #




