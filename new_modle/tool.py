import os
import tensorflow as tf
from PIL import Image


# img_path='/Users/wywy/Desktop/客观题分类/7'
# save_path='/Users/wywy/Desktop/客观题分类/67'
#
# for file in os.listdir(img_path):
#     if file=='.DS_Store':
#         os.remove(img_path+'/'+file)
#     else:
#         img=Image.open(img_path+'/'+file)
#         img.save(save_path+'/'+file)



# img_path='/Users/wywy/Desktop/客观题分类/7'
# save_path='/Users/wywy/Desktop/客观题分类/test_7'
# count=0
#
# for file in os.listdir(img_path):
#     if file=='.DS_Store':
#         os.remove(img_path+'/'+file)
#     else:
#         if count < 6000:
#             img=Image.open(img_path+'/'+file)
#             img.save(save_path+'/'+file)
#             os.remove(img_path+'/'+file)
#             count+=1
# print(count)

# img_path='/Users/wywy/Desktop/客观题分类/2'
# for file in os.listdir(img_path):
#     if file=='.DS_Store':
#         os.remove(img_path)
#     else:
#         name=file.split('.')[0]
#         print(name)

# img_path='/Users/wywy/Desktop/客观题分类/test_7'
# # '/Users/wywy/Desktop/客观题分类/test_0'
# # /Users/wywy/Desktop/训练完成模型/客观题数据（单、多选）
# save_path='/Users/wywy/Desktop/客观题分类/多分类/test_img'
#
# count=19564
# for file in os.listdir(img_path):
#     if file=='.DS_Store':
#         os.remove(img_path+'/'+file)
#     else:
#         img=Image.open(img_path+'/'+file)
#         img.save(save_path+'/'+str(count)+'_7.jpg')
#         count+=1
# print(count)

#塞选色度较暗数据
import os
import cv2

import scipy.misc

#
# img_path='/Users/wywy/Desktop/客观题分类/多分类/train_img'
# save_path1='/Users/wywy/Desktop/客观题分类/多分类/train1_img'
# # save_path2='/Users/wywy/Desktop/训练完成模型/客观题数据（单、多选）/train3_img'
#
# def img_aug(img_path,save_path):
#     for file in os.listdir(img_path):
#         if file=='.DS_Store':
#             os.remove(img_path+'/'+file)
#         else:
#             img=cv2.imread(img_path+'/'+file)
#             img1=img
#             count=0
#             for i in range(img.shape[0]):
#                 for j in range(img.shape[1]):
#                     # img1[i,j]=0.2 * img[i,j][0] + 0.25 * img[i,j][1] +  0.22 * img[i,j][2]    #阈值自己调节
#                     if img1[i,j][0]<70 and img1[i,j][1]<70 and img1[i,j][1]<70:
#                         count+=1
#             if count>168*16/8:
#                 scipy.misc.imsave(save_path+'/'+file, img1)
# img_aug(img_path,save_path1)






#删除色度较暗数据
# img_path='/Users/wywy/Desktop/remove_img'
# save_path='/Users/wywy/Desktop/客观题分类/多分类/train_img'
#
# count=0
# for file in os.listdir(img_path):
#     if file=='.DS_Store':
#         os.remove(img_path+'/'+file)
#     else:
#         os.remove(save_path+'/'+file)
#         # print(save_path+'/'+file)
#         count+=1
# print(count)



# img_path='/Users/wywy/Desktop/客观题分类/0'
#
# for file in os.listdir(img_path):
#     if file=='.DS_Store':
#         os.remove(img_path+'/'+file)
#     else:
#         name=file.split('.')[0].split('_')
#         print(name)
#         print(len(name))


# img_path='/Users/wywy/Desktop/训练完成模型/客观题数据（单、多选）/test1_img'
# save_path='/Users/wywy/Desktop/客观题分类/cls_test'
# count=0
# for file in os.listdir(img_path):
#     if file=='.DS_Store':
#         os.remove(img_path+'/'+file)
#     else:
#         img=Image.open(img_path+'/'+file)
#         img.save(save_path+'/'+str(count)+'_1.jpg')
#         count+=1
# print(count)

# img_path='/Users/wywy/Desktop/test11'
# save_path='/Users/wywy/Desktop/mobile_test'
# count=0
# for file in os.listdir(img_path):
#     if file=='.DS_Store':
#         os.remove(img_path+'/'+file)
#     else:
#         bg=Image.new('RGB',(168,16),'white')
#         img=Image.open(img_path+'/'+file).convert('RGB')
#         img=img.resize((int(168/7*4),16),Image.ANTIALIAS)
#         bg.paste(img,(0,0))
#         bg.save(save_path+'/'+str(count)+'.jpg')
#         count+=1













