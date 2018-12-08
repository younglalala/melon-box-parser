import os
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as iming
from PIL import  Image ,ImageFilter,ImageDraw
import Augmentor
import scipy.misc
import scipy.signal
import scipy.ndimage


img_path='/Users/wywy/Desktop/XX'
save_path='/Users/wywy/Desktop/cnn_train_2'


#截取图片
def cut_img(img_path,save_path):
    for file in os.listdir(img_path):
        im = Image.open(img_path+'/'+file)
        region=im.crop((6,6,64-5,64-5))
        region.save(save_path+'/'+file)


#变换图片大小（resize）
def resize_picture(img_path):
    for i in os.listdir(img_path):
        if i=='.DS_Store':
            os.remove(img_path+'/'+i)
        else:
            im=Image.open(img_path+'/'+i)
            out=im.resize((64,318),Image.ANTIALIAS)
            out.save(img_path+'/'+i)
# resize_picture(img_path)

#图片分类：
def pictuer_clas(img_path):
    for file in os.listdir(img_path):
        if file == '.DS_Store':
            os.remove(img_path + '/' + file)
        else:
            name1=file.split('.')[0].split('_')[-1]

            # if len(list(name1))==1:
            #     name=name1
            # else:
            #     name=list(name1[0])
            if name1=='9':
                im=Image.open(img_path+'/'+file)
                # print(save_path+file)
                im.save(save_path+file)
# pictuer_clas(img_path)


#更改文件路径
def change_path(img_path,save_path):
    for file in os.listdir(img_path):
        if file=='.DS_Store':
            os.remove(img_path+'/'+file)
        else:
            im=Image.open(img_path+'/'+file)

            im.save(save_path+'/'+file)

change_path(img_path,save_path)

#在所有样本中找出原始数据
# def fin
# for file in os.listdir(save_path):
#     if file=='.DS_Store':
#         os.remove(save_path+'/'+file)
#     else:
#         name = file.split('.')[0].split('_')[0]
#         if int(name) >=30000:
#             os.remove(save_path+'/'+file)


#数据增强(扭曲旋转)
def picture_aug(img_path):
    p = Augmentor.Pipeline(img_path)
    p.random_distortion(probability=0.5, grid_width=1, grid_height=1, magnitude=1)
    p.rotate(probability=0.5, max_left_rotation=10, max_right_rotation=10)
    p.sample(40000)


#数据增强（增加噪点，镜像）
def salt(img_path,n,save_path):
    for file in os.listdir(img_path):
        if file=='.DS_Store':
            os.remove(img_path+'/'+file)
        else:
            img=cv2.imread(img_path+'/'+file)
            for k in range(n):
                # 随机选择椒盐的坐标
                i = int(np.random.random() * img.shape[1])
                j = int(np.random.random() * img.shape[0])
                # 如果是灰度图
                if img.ndim == 2:
                    img[j, i] = 255
                    # 如果是RBG图片
                elif img.ndim == 3:
                    img[j, i, 0] = 255
                    img[j, i, 1] = 255
                    img[j, i, 2] = 255
            scipy.misc.imsave(save_path+'/'+'aug1_'+file, img)
# salt(img_path,1000,save_path)


#rename
def rename(img_path):
    xx=6475
    for file in os.listdir(img_path):
        if file == '.DS_Store':
            os.remove(img_path + '/' + file)
        else:
            # name1 = file.split('.')[0].split('_')[-1]
            im=Image.open(img_path+'/'+file)
            # # print('/Users/wywy/Desktop/flase_img/{}_{}.jpg'.format(xx,name1))
            im.save('/Users/wywy/Desktop/XX/{}_X.jpg'.format(xx))
            # os.remove(img_path+'/'+file)
            xx+=1
    print(xx)
# rename(img_path)

# for i in os.listdir(img_path):
#
#     img = Image.open(img_path+'/'+i)
#     img_trans1 = img.transpose(Image.FLIP_LEFT_RIGHT)   #图片进行镜像处理。
#     img_trans1.save(img_save_path + '/' + 'trans1' + str(i))
    # print(img_save_path + '/' + 'trans1' + str(i))
    # img_trans2 = img.transpose(Image.FLIP_TOP_BOTTOM)   #图片进行翻转处理。
    # img_trans2.save(img_save_path+'/'+'trans2'+str(i))

#知识点，暂时未用到。
#增加噪点
# def medium_filter(im, x, y, step):
#     sum_s = []
#     for k in range(-int(step / 2), int(step / 2) + 1):
#         for m in range(-int(step / 2), int(step / 2) + 1):
#             sum_s.append(im[x + k][y + m])
#     sum_s.sort()
#     return sum_s[(int(step * step / 2) + 1)]
#
#
# def mean_filter(im, x, y, step):
#     sum_s = 0
#     for k in range(-int(step / 2), int(step / 2) + 1):
#         for m in range(-int(step / 2), int(step / 2) + 1):
#             sum_s += im[x + k][y + m] / (step * step)
#     return sum_s
#
#
# def convert_2d(r):
#     n = 3
#     # 3*3 滤波器, 每个系数都是 1/9
#     window = np.ones((n, n)) / n ** 2
#     # 使用滤波器卷积图像
#     # mode = same 表示输出尺寸等于输入尺寸
#     # boundary 表示采用对称边界条件处理图像边缘
#     s = scipy.signal.convolve2d(r, window, mode='same', boundary='symm')
#     return s.astype(np.uint8)
#
#
# # def convert_3d(r):
# #     s_dsplit = []
# #     for d in range(r.shape[2]):
# #         rr = r[:, :, d]
# #         ss = convert_2d(rr)
# #         s_dsplit.append(ss)
# #     s = np.dstack(s_dsplit)
# #     return s
#
#
# def add_salt_noise(img):
#     rows, cols, dims = img.shape
#     R = np.mat(img[:, :, 0])
#     G = np.mat(img[:, :, 1])
#     B = np.mat(img[:, :, 2])
#
#     Grey_sp = R * 0.299 + G * 0.587 + B * 0.114
#     Grey_gs = R * 0.299 + G * 0.587 + B * 0.114
#
#     snr = 0.9
#     mu = 0
#     sigma = 0.12
#
#     noise_num = int((1 - snr) * rows * cols)
#
#     for i in range(noise_num):
#         rand_x = random.randint(0, rows - 1)
#         rand_y = random.randint(0, cols - 1)
#         if random.randint(0, 1) == 0:
#             Grey_sp[rand_x, rand_y] = 0
#         else:
#             Grey_sp[rand_x, rand_y] = 255
#
#     Grey_gs = Grey_gs + np.random.normal(0, 48, Grey_gs.shape)
#     Grey_gs = Grey_gs - np.full(Grey_gs.shape, np.min(Grey_gs))
#     Grey_gs = Grey_gs * 255 / np.max(Grey_gs)
#     Grey_gs = Grey_gs.astype(np.uint8)
#
#     # 中值滤波
#     Grey_sp_mf = scipy.ndimage.median_filter(Grey_sp, (8, 8))
#     Grey_gs_mf = scipy.ndimage.median_filter(Grey_gs, (8, 8))
#
#     # 均值滤波
#     n = 3
#     window = np.ones((n, n)) / n ** 2
#     Grey_sp_me = convert_2d(Grey_sp)
#     Grey_gs_me = convert_2d(Grey_gs)
#
#     plt.subplot(321)
#     plt.title('Grey salt and pepper noise')
#     plt.imshow(Grey_sp, cmap='gray')
#     plt.subplot(322)
#     plt.title('Grey gauss noise')
#     plt.imshow(Grey_gs, cmap='gray')
#
#     plt.subplot(323)
#     plt.title('Grey salt and pepper noise (medium)')
#     plt.imshow(Grey_sp_mf, cmap='gray')
#     plt.subplot(324)
#     plt.title('Grey gauss noise (medium)')
#     plt.imshow(Grey_gs_mf, cmap='gray')
#
#     plt.subplot(325)
#     plt.title('Grey salt and pepper noise (mean)')
#     plt.imshow(Grey_sp_me, cmap='gray')
#     plt.subplot(326)
#     plt.title('Grey gauss noise (mean)')
#     plt.imshow(Grey_gs_me, cmap='gray')
#     plt.show()

#
#
#
# #模糊图片
# base_path = "../data-set"
# flag=0
# for i in os.listdir(base_path):
#     flag+=1
#     name=i.split('.')[0].split('_')[1]
#     rename=str(flag)+'_'+name+'.jpg'
#     im=np.array(Image.open(base_path+'/'+i))
#     add_salt_noise(im)
#     im2=im.filter(ImageFilter.BLUR)
#     im2.save(base_path+'/'+rename)
#
# #扭曲
# import Augmentor
# p = Augmentor.Pipeline("/path/to/images")
#
# p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
# p.zoom(probability=0.5, min_factor=1.1, max_factor=1.5)
#
# p.sample(10000)
# p.process()
# p.sample(100, multi_threaded=False)
# p = Augmentor.Pipeline("/path/to/images")
#
#
# # Point to a directory containing ground truth data.
# # Images with the same file names will be added as ground truth data
# # and augmented in parallel to the original data.
# p.ground_truth("/path/to/ground_truth_images")
# # Add operations to the pipeline as normal:
# p.rotate(probability=1, max_left_rotation=5, max_right_rotation=5)
# p.flip_left_right(probability=0.5)
# p.zoom_random(probability=0.5, percentage_area=0.8)
# p.flip_top_bottom(probability=0.5)
# p.sample(50)
#
#
#
# g = p.keras_generator(batch_size=128)
# images, labels = next(g)
#
# import torchvision
# transforms = torchvision.transforms.Compose([
#     p.torch_transform(),
#     torchvision.transforms.ToTensor(),
# ])
# p.random_distortion(probability=1, grid_width=4, grid_height=4, magnitude=8)
# p.flip_left_right(probability=0.5)
# p.flip_top_bottom(probability=0.5)
# p.sample(100)
#
# import Augmentor
#
# p = Augmentor.Pipeline("/home/user/augmentor_data_tests")
#
# p.rotate90(probability=0.5)
# p.rotate270(probability=0.5)
# p.flip_left_right(probability=0.8)
# p.flip_top_bottom(probability=0.3)
# p.crop_random(probability=1, percentage_area=0.5)
# p.resize(probability=1.0, width=120, height=120)





