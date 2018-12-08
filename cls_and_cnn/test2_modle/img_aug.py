
import cv2
import os
from PIL import Image

import scipy.misc


# #根据图像调节
# img_path='/Users/wywy/Desktop/未填涂选项'
# img_save='/Users/wywy/Desktop/aug00'
def img_aug(img_path,save_path):
    for file in os.listdir(img_path):
        if file=='.DS_Store':
            os.remove(img_path+'/'+file)
        else:
            img=cv2.imread(img_path+'/'+file)
            img1=img
            # name=file.split('.')[0].split('-')
            # name0=name[0]
            # name1=list(name[1])[2]
            # if name1=='X':
            #     pass
            for i in range(img.shape[0]):
                for j in range(92):
                    img1[i,j]=0.2 * img[i,j][0] + 0.25 * img[i,j][1] +  0.22 * img[i,j][2]    #阈值自己调节
            # print(save_path+'/''aug0'+name0+'_'+name1+'.jpg')
            scipy.misc.imsave(save_path+'/'+file, img)
# img_aug(img_path,img_save)




# # resize
# img_path='/Users/wywy/Desktop/未填涂选项'
# save_path='/Users/wywy/Desktop/train1_img'
def resize(img_path,save_path):
    flag=7640
    for file in os.listdir(img_path):
        if file=='.DS_Store':
            os.remove(img_path+'/'+file)
        else:
            img=Image.open(img_path+'/'+file)
            # out = img.resize((168, 16), Image.ANTIALIAS)
            # print(img_path + '/' + file)
            # print(save_path + '/' + str(flag)+'_X.jpg')
            img.save(save_path + '/' + str(flag)+'_X.jpg')
            flag+=1
    print(flag)
# resize(img_path,save_path)
#
# img_path='/Users/wywy/Desktop/train1_img'
# flag=0
# for file in os.listdir(img_path):
#     if file == '.DS_Store':
#         os.remove(img_path + '/' + file)
#     else:
#         name=file.split('.')[0].split('_')[-1]
#         if name=='X':
#             os.remove(img_path+'/'+file)
#             flag+=1
# print(flag)






# def rename_resize(img_path,save_path):
#     for file in os.listdir(img_path):
#         if file=='.DS_Store':
#             os.remove(img_path+'/'+file)
#         else:
#             img = Image.open(img_path + '/' + file)
#             out = img.resize((168, 16), Image.ANTIALIAS)
#             # print(img_path + '/' + file)
#             out.save(save_path + '/' + file)







