import os
from  PIL import Image
import numpy as np
import random
import Augmentor
import cv2 as cv
import scipy.misc
# test_id=935
# img_path='/Users/wywy/Desktop/img_data/parse'
# save_path='/Users/wywy/Desktop/train1_img'
# save_path='/Users/wywy/Desktop/train2_img'


#resize  and  rename
def resize_name(img_path,save_path):
    '''
    :param img_path: original image path
    :param save_path: save path
    :return: None
    '''
    flag=0    #最后一次数据。947
    for file in os.listdir(img_path):
        if file=='.DS_Store':
            os.remove(img_path+'/'+file)
        else:
            name= list(file.split('.')[0].split('-')[-1])
            img = Image.open(img_path + '/' + file)
            out=img.resize((168, 16), Image.ANTIALIAS)
            if len(name)  !=6:
                print(file)
                os.remove(img_path+'/'+file)
            else:
                # print(save_path+'/'+'aug10'+str(flag)+'_'+name[2]+'.jpg')
                out.save(save_path+'/'+'aug20_'+str(flag)+'_'+name[2]+'.jpg')
                flag+=1
    print(flag)


#更改文件路径
def img_remove(img_path,save_path):
    '''change image path'''
    flag=0
    for file in os.listdir(img_path):
        if file=='.DS_Store':
            os.remove(img_path+'/'+file)
        else:
            img=Image.open(img_path+'/'+file)
            img.save(save_path+'/'+file)
            flag+=1
    print(flag)




#选项图片增强
# img_path='/Users/wywy/Desktop/选项'
def picture_aug(img_path):
    p = Augmentor.Pipeline(img_path)
    p.random_distortion(probability=0.5, grid_width=2, grid_height=2, magnitude=2)
    p.rotate(probability=0.5, max_left_rotation=5, max_right_rotation=5)
    p.sample(100)
# picture_aug(img_path)


def paste_img(img_path1,img_path2,save_path):
    all_img=[]
    location=[(4,0),(26,0),(48,0),(70,0)]
    for file in os.listdir(img_path1):
        if file=='.DS_Store':
            os.remove(img_path1+'/'+file)
        else:
            img=Image.open(img_path1+'/'+file)
            all_img.append(img)
    flag=7000
    for file2 in os.listdir(img_path2):
        if file2=='.DS_Store':
            os.remove(img_path2+'/'+file2)
        else:
            name = list(file2.split('.')[0].split('-')[-1])
            if len(name)  !=6:
                os.remove(img_path2+'/'+file2)
            else:
                dict_infor={0:'A',1:'B',2:'C',3:'D'}
                img2=Image.open(img_path2+'/'+file2)
                name1=file2.split('.')[0].split('-')
                name2=list(name1[-1])[2]
                if name2=='X':
                    os.remove(img_path2+'/'+file2)
                name0=name1[0]
                index=np.random.randint(len((all_img)))
                xx = np.random.randint(len(location))
                if name[2]=='A':
                    if xx==0:
                        img2.paste(all_img[index], location[3])
                        img2.save(save_path + '/' +name0+'_'+name2+'_'+dict_infor.get(3)+'.jpg')
                    else:
                        img2.paste(all_img[index],location[xx])
                        img2.save(save_path + '/' + name0 + '_' + name2 + '_' + dict_infor.get(xx) + '.jpg')

                elif name[2]=='B':
                    if xx == 1:
                        img2.paste(all_img[index], location[2])
                        img2.save(save_path + '/' + name0 + '_' + name2 + '_' + dict_infor.get(2) + '.jpg')
                    else:
                        img2.paste(all_img[index], location[xx])
                        img2.save(save_path + '/' + name0 + '_' + name2 + '_' + dict_infor.get(xx) + '.jpg')
                elif name[2]== 'C':
                    if xx == 2:
                        xx=0
                        img2.paste(all_img[index], location[0])
                        img2.save(save_path + '/' + name0 + '_' + name2 + '_' + dict_infor.get(0) + '.jpg')
                    else:
                        img2.paste(all_img[index], location[xx])
                        img2.save(save_path + '/' + name0 + '_' + name2 + '_' + dict_infor.get(xx) + '.jpg')

                else:
                    img2.paste(all_img[index], location[2])
                    img2.save(save_path + '/' + name0 + '_' + name2 + '_' + dict_infor.get(2) + '.jpg')
                # if xx==0:
                #     img2.save(save_path + '/' +str(flag)+'_'+name[2]+'_A'+'.jpg')
                # elif xx==1:
                #     img2.save(save_path + '/' + str(flag) + '_' + name[2] + '_B' + '.jpg')
                # elif xx==2:
                #     img2.save(save_path + '/' + str(flag) + '_' +name[2]  + '_C' + '.jpg')
                # else:
                #     img2.save(save_path + '/' + str(flag) + '_' + name[2] + '_D' + '.jpg')
                # flag+=1
# paste_img(img_path1,img_path2,save_path)

#粘贴到三个选项

def img_paste3(img_path1,img_path2,save_path):
    all_img=[]
    location = [(92+22+22, 0)]
    for file in os.listdir(img_path1):
        if file=='.DS_Store':
            os.remove(img_path1+'/'+file)
        else:
            img=Image.open(img_path1+'/'+file)
            all_img.append(img)
    flag=0
    for file2 in os.listdir(img_path2):
        if file2=='.DS_Store':
            os.remove(img_path2+'/'+file2)
        else:
            index = np.random.randint(len((all_img)))
            name=file2.split('.')[0]
            img2=Image.open(img_path2+'/'+file2)

            img2.paste(all_img[index], location[0])
            # print(save_path + '/' +name+'_E.jpg')
            img2.save(save_path + '/' +name+'_G.jpg')
            flag+=1




# img_path='/Users/wywy/Desktop/七个选项空白'
# img1_path='/Users/wywy/Desktop/选项'
# save_path='/Users/wywy/Desktop/客观题分类/0'
# all_paste=[]
# for file in os.listdir(img1_path):
#     if file=='.DS_Store':
#         os.remove(img1_path+'/'+file)
#     else:
#         paste_image=Image.open(img1_path+'/'+file)
#         all_paste.append(paste_image)
# random.shuffle(all_paste)
#
# # all_bg=[]
# sett10=[]
# for mm in range(1000):
#     sett1 = []
#     for m in range(1000):
#         if len(set(sett1)) < 0:
#             index2=random.randint(0,6)
#             sett1.append(index2)
#     sett10.append(list(set(sett1)))
#
# name_dict = dict(zip([0, 1, 2, 3, 4, 5, 6], list('ABCDEFG')))
#
# sett = [(6, 0), (30, 0), (54, 0), (78, 0), (102, 0), (126, 0), (150, 0)]
# count=13200
# for mm in range(100):
#     for file1 in os.listdir(img_path):
#
#         if file1=='.DS_Store':
#             os.remove(img_path+'/'+file1)
#         else:
#             bg=Image.open(img_path+'/'+file1)
#             set_index = random.randint(0, len(sett10) - 1)
#             img_name = ''
#             for jj in sett10[set_index]:
#                 index = random.randint(0, len(all_paste) - 1)
#                 bg.paste(all_paste[index], sett[int(jj)])
#                 img_name += '_'+name_dict.get(int(int(jj)))
#             # bg.save(save_path + '/add' + str(count) + img_name + '.jpg')   有填涂的时候的保存方式
#             bg.save(save_path + '/add' + str(count) + '_X' + '.jpg')   #没有填涂的时候的保存方式
#             count += 1
# print(count)


    #     if len(set(sett1))<6:
    #
    #         index2=random.randint(0,6)
    #         sett1.append(index2)
    # print(sett1)



# count=0

# for i in sett10:
#     img_name = ''
#     bg_index = random.randint(0, len(all_bg) - 1)
#     bgg = all_bg[bg_index]
#     for ii in i:
#         index = random.randint(0, len(all_paste) - 1)
#         bgg.paste(all_paste[index],sett[ii])
#         print(ii)
#
#         img_name+=name_dict.get(int(ii))
#
#
#
#         bgg.save(save_path+'/'+str(count)+'_'+img_name+'.jpg')
#         count+=1


# count=0
#
# for i in range(3):
#     bg_index=random.randint(0,len(all_bg)-1)
#     set_index=random.randint(0,len(sett10)-1)
#     bgg = all_bg[bg_index]
#
#     img_name = ''
#     for jj in sett10[set_index]:
#         index = random.randint(0, len(all_paste) - 1)
#         bgg.paste(all_paste[index],sett[int(jj)])
#         print(all_paste[index],sett[int(jj)])
#         print(bgg)
#
#         img_name+=name_dict.get(int(int(jj)))
#     print('-------------------')
#     bgg.save(save_path + '/' + str(count) + '_' + img_name + '.jpg')
#     count+=


def img_aug(img_path,save_path):
    count=0
    for i in range(20):
        all_file=[]

        for file in os.listdir(img_path):
            if file=='.DS_Store':
                os.remove(img_path+'/'+file)
            else:
                all_file.append(file)
        random.shuffle(all_file)
        for f in all_file:
            if count<5000:
                img=cv.imread(img_path+'/'+f)
                img1=img
                aa=[0.68,0.65,0.67,0.7,0.69,0.66,0.64,0.63,0.61,0.62,0.6,0.8]
                for i in range(img.shape[0]):
                    for j in range(img.shape[1]):
                        index=random.randint(0,len(aa)-1)
                        img1[i, j] =img1[i,j][0]*aa[index],img1[i,j][1]*aa[index],img1[i,j][2]*aa[index]
                scipy.misc.imsave(save_path+'/mm'+str(count)+f, img1)
                # print(save_path+'/aug'+f)

                count+=1
    print(count)



#选项分类：
def cls_ob(img_path,save_path,choice_set):
    '''
    把所有单选题数据中不同选项个数的分开以便后期生成多选题
    :param img_path: original image path
    :param save_path: cls save path
    :param choice_set: the set of image area
    :return:None
    '''
    count=0
    for file in os.listdir(img_path):
        if file=='.DS_Store':
            os.remove(img_path+'/'+file)
        else:
            img = cv.imread(img_path + '/' + file)
            img1=img
            cc=0

            for i in range(img.shape[0]):
                for j in range(choice_set,img.shape[1]):
                    # print(img1[i,j][0]<230,img1[i,j][1]<230,img1[i,j][2]<230)
                    if img1[i,j][0]<230 and img1[i,j][1]<230 and img1[i,j][2]<230:
                        cc+=1
            if cc<5:
                scipy.misc.imsave(save_path+'/'+file, img1)
                count+=1
    print(count)

# choice_set=[(0,0,22,16),(22,0,47,16),(47,0,72,16),(72,0,94,16),(94,0,118,16),(118,0,142,16),(142,0,168,16)]
# img_path='/Users/wywy/Desktop/客观单选模型9:26/data/o_train'
# save_path='/Users/wywy/Desktop/c_cls/7'
# cls_ob(img_path,save_path,168)
# c=0


#图像调整明暗度
def dark_img(img_path,save_path,dark_value,image_set,save_name):
    '''
    调整图片的明暗程度，以达到手机数据的效果
    :param img_path: original image path
    :param save_path:  save image path
    :param dark_value: the degree of dark ,type :float
    :param image_set: image dark area , type: int
    :param save_name: image save name ,type :string
    :return: None
    '''
    count=0
    all_file=[]
    for file in os.listdir(img_path):
        if file=='.DS_Store':
            os.remove(img_path+'/'+file)
        else:
            all_file.append(file)
    random.shuffle(all_file)
    for f in all_file:
        img=cv.imread(img_path+'/'+f)
        img1=img
        for i in range(img.shape[0]):
            for j in range(image_set):
                img1[i, j] =img1[i,j][0]*dark_value,img1[i,j][1]*dark_value,img1[i,j][2]*dark_value
        scipy.misc.imsave(save_path+'/'+str(count)+'_'+save_name+'.jpg', img1)
        count+=1
    print(count)





#4/5个空白选项生成6/7个空白选项
def generate_choice(img_path,img_path2,save_path):
    '''
    通过4/5个空白选项来生成6/7个空白选项
    :param img_path: 4/5个空白选项
    :param img_path2: 截取出来的单独的空白选项，size（16，16）
    :param save_path: 保存路径
    :return: None
    '''
    choice_set=[(6,0),(30,0),(54,0),(78,0),(102,0),(126,0),(150,0)]
    choice_dict=dict(zip(choice_set,list('ABCDEFG')))

    all_paste=[]
    for file in os.listdir(img_path2):
        if file == '.DS_Store':
            os.remove(img_path2+'/'+file)
        else:
            p_img = Image.open(img_path2+'/'+file)
            all_paste.append(p_img)
    random.shuffle(all_paste)
    c=0
    for i in range(200):
        for f in os.listdir(img_path):
            if f=='.DS_Store':
                os.remove(img_path+'/'+f)
            else:
                img=Image.open(img_path+'/'+f)
                img.paste(random.sample(all_paste,1)[0],(random.sample(choice_set,1)[0]))

                img.save(save_path+'/'+str(c)+'_X.jpg')
                c+=1
    print(c)

#6/7个空白选项生成6/7单选题

def gen_img(img_path,img_path2,save_path):
    choice_set=[(6,0),(30,0),(54,0),(78,0),(102,0),(126,0),(150,0)]
    choice_dict=dict(zip(choice_set,list('ABCDEFG')))

    all_paste=[]
    for file in os.listdir(img_path2):
        if file == '.DS_Store':
            os.remove(img_path2+'/'+file)
        else:
            p_img = Image.open(img_path2+'/'+file)
            all_paste.append(p_img)
    random.shuffle(all_paste)

    c=7000

    for f in os.listdir(img_path):
        if f=='.DS_Store':
            os.remove(img_path+'/'+f)
        else:
            img=Image.open(img_path+'/'+f)
            random_set=random.sample(choice_set[:-1], 1)[0]
            print(random_set,c)
            print(choice_dict.get(random_set),c)
            img.paste(random.sample(all_paste, 1)[0], (random_set))
            img.save(save_path+'/'+str(c)+'_'+choice_dict.get(random_set)+'.jpg' )
            # print(save_path+'/'+str(c)+'_'+choice_dict.get(random_set)+'.jpg' )
            c+=1
    print(c)




#--------------------------------------------
#数据预处理

# img_path='/Users/wywy/Desktop/客观题数据/data/ok_img/验证/935/object-box'
# # img_path='/Users/wywy/Desktop/客观题数据/data/ok_img/训练/947'
# save_path='/Users/wywy/Desktop/客观题数据/data/ok_img/validation_img'
#
# cc=6720
# for file in os.listdir(img_path):
#     if file=='.DS_Store':
#         os.remove(img_path+'/'+file)
#     else:
#         name= list(file.split('.')[0].split('-')[-1])
#         img=Image.open(img_path+'/'+file)
#         if len(name)==6:
#             label=name[2]
#             if label=='A':
#                 img.save(save_path+'/A/'+str(cc)+'_'+label+'.jpg')
#             elif label == 'B':
#                 img.save(save_path + '/B/' + str(cc) + '_' + label + '.jpg')
#
#             elif label == 'C':
#                 img.save(save_path + '/C/' + str(cc) + '_' + label + '.jpg')
#
#             elif label == 'D':
#                 img.save(save_path + '/D/' + str(cc) + '_' + label + '.jpg')
#
#             elif label == 'E':
#                 img.save(save_path + '/E/' + str(cc) + '_' + label + '.jpg')
#
#             elif label == 'F':
#                 img.save(save_path + '/F/' + str(cc) + '_' + label + '.jpg')
#
#             elif label == 'G':
#                 img.save(save_path + '/G/' + str(cc) + '_' + label + '.jpg')
#
#             elif label == 'X':
#                 img.save(save_path + '/X/' + str(cc) + '_' + label + '.jpg')
#             cc+=1
# print(cc)

# sett = [(6, 0), (30, 0), (54, 0), (78, 0), (102, 0), (126, 0), (150, 0)]
#
# for file in os.listdir(img_path):
#     if file=='.DS_Store':
#         os.remove(img_path+'/'+file)
#     else:
# img_path='/Users/wywy/Desktop/七个选项空白/空白选项'
# for file in os.listdir(img_path):
#     if file=='.DS_Store':
#         os.remove(img_path+'/'+file)
#     else:
#         img=Image.open(img_path+'/'+file)
#         img=img.resize((16,16),Image.ANTIALIAS)
#         img.save(img_path+'/'+file)



#生成单选FG数据-------------------------------------


#选项resize
def resize_chioce(img_path):
    for file in os.listdir(img_path):
        if file=='.DS_Store':
            os.remove(img_path+'/'+file)
        else:
            img=Image.open(img_path+'/'+file)
            out=img.resize((16,16),Image.ANTIALIAS)
            out.save(img_path+'/'+file)

len_chioce=7
choice_num=7
img_path2='/Users/wywy/Desktop/选项深色'
img_path='/Users/wywy/Desktop/c_cls/{}'.format(len_chioce)
save_path='/Users/wywy/Desktop/c_cls/{}个选项'.format(choice_num)

def gen_mchoice(len_chioce,choice_num,choice_path,orig_path,save_path):


    sett = [(0, 0), (25, 0), (50, 0), (75, 0), (100, 0), (125, 0), (150, 0)]
    all_pp=[]
    for file1 in os.listdir(img_path2):
        if file1=='.DS_Store':
            os.remove(img_path2+'/'+file1)
        else:
            all_pp.append(file1)
    random.shuffle(all_pp)

    cc=0
    for i in range(10):
        all_file=[]
        for file in os.listdir(img_path):
            if file=='.DS_Store':
                os.remove(img_path+'/'+file)
            else:
                all_file.append(file)
        random.shuffle(all_file)
        for f in all_file:
            if cc<0+50000:
                new_setdict=dict(zip([3,4,5,6,7],[[(0, 0), (25, 0), (50, 0)],
                                                  [(0, 0), (25, 0), (50, 0), (75, 0)],
                                                  [(0, 0), (25, 0), (50, 0), (75, 0), (100, 0)],
                                                  [(0, 0), (25, 0), (50, 0), (75, 0), (100, 0), (125, 0)],
                                                  [(0, 0), (25, 0), (50, 0), (75, 0), (100, 0), (125, 0), (150, 0)]]))
                new_set=new_setdict.get(len_chioce)
                chioce_num=choice_num-1
                img=Image.open(img_path+'/'+f).resize((168,16))
                f_name=f.split('.')[0].split('_')[-1]
                index_dict1=dict(zip(list('ABCDEFG'),sett))
                index_dict2 = dict(zip(sett, list('ABCDEFG')))
                if index_dict1.get(f_name) in new_set:
                    new_set.remove(index_dict1.get(f_name))
                randomlist = random.sample(new_set[:], chioce_num)
                for ss in randomlist:
                    name=index_dict2.get(ss)
                    p_img=Image.open(img_path2+'/'+random.sample(all_pp,1)[0])
                    img.paste(p_img,ss)
                    f_name+='_'+name
                img.save(save_path+'/'+str(cc)+'_'+f_name+'.jpg')
                cc+=1
    print(cc)








# # ###################################################################
# img_path='/Users/wywy/Desktop/手机扫描样例1'
# cc=0
# for file in os.listdir(img_path):
#     if file=='.DS_Store':
#         os.remove(img_path+'/'+file)
#     else:
#         bg=Image.new('RGB',(168,16),'white')
#         img=Image.open(img_path+'/'+file)
#         img=img.resize((int(168/7*7),16),Image.ANTIALIAS)
#         bg.paste(img,(0,0))
#         bg.save(img_path+'/'+str(cc)+'.jpg')
#         cc+=1
# #





# img_path='/Users/wywy/Desktop/客观题数据/data/ok_img/test_img/all_test'
# save_path='/Users/wywy/Desktop/客观题数据/data/ok_img/test_img'
# cc=0
# for file in os.listdir(img_path):
#     if file=='.DS_Store':
#         os.remove(img_path+'/'+file)
#     else:
#         name=file.split('.')[0].split('_')
#         img=Image.open(img_path+'/'+file)
#         if len(name)==2:
#             if name[-1]=='X':
#                 img.save(save_path+'/test_cls/'+str(cc)+'_0.jpg')
#             else:
#                 img.save(save_path+'/test_cls/'+str(cc)+'_1.jpg')
#         elif len(name)==3:
#             img.save(save_path+'/test_cls/'+str(cc)+'_2.jpg')
#         elif len(name)==4:
#             img.save(save_path+'/test_cls/'+str(cc)+'_3.jpg')
#         elif len(name)==5:
#             img.save(save_path + '/test_cls/' +str(cc)+'_4.jpg')
#
#         elif len(name)==6:
#             img.save(save_path + '/test_cls/' +str(cc)+'_5.jpg')
#         elif len(name)==7:
#             img.save(save_path + '/v_cls/' +str(cc)+'_6.jpg')
#         elif len(name)==8:
#             img.save(save_path + '/test_cls/' +str(cc)+'_7.jpg')
#         cc+=1




# img_path='/Users/wywy/Desktop/手机扫描样例'

# for file in os.listdir(img_path):
#     if file=='.DS_Store':
#         os.remove(img_path+'/'+file)
#     else:
#         set=96/3
#         all_c=[]
#         img = cv.imread(img_path + '/' + file)
#         for i in range(3):
#             img1=img
#             cc=0
#             for ii in range(img.shape[0]):
#                 for j in range(int(set*(i)),int(set*(i+1))):
#                     # img1[i, j] =img1[i,j][0]*0.9,img1[i,j][1]*0.9,img1[i,j][2]*0.9
#                     if img1[ii,j][0]<100 and img1[ii,j][1]<100 and img1[ii,j][2]<100:
#                         cc+=1
#             all_c.append(cc)
#         print(all_c)




#图片增加椒盐噪点
# img_path='/Users/wywy/Desktop/cls_train'
# save_path='/Users/wywy/Desktop/train_aug'
# all_file=[]
# for file in os.listdir(img_path):
#     if file=='.DS_Store':
#         os.remove(img_path+'/'+file)
#     else:
#         all_file.append(file)
# random.shuffle(all_file)
# c=0
# for f in all_file:
#     if c<25000:
#         img=cv.imread(img_path+'/'+f)
#         count=200
#         for k in range(0,count):
#             xi = int(np.random.uniform(0, img.shape[1]))
#             xj = int(np.random.uniform(0, img.shape[0]))
#             # add noise
#             if img.ndim == 2:
#                 img[xj, xi] = 255
#             elif img.ndim == 3:
#                 img[xj, xi, 0] = 255
#                 img[xj, xi, 1] = 255
#                 img[xj, xi, 2] = 255
#         scipy.misc.imsave(save_path+'/b'+f,img)
#         c+=1
# print(c)
#
