# 二值化图像算法

from PIL import Image

def binarizing(img,threshold): #input: gray image
    pixdata = img.load()
    w, h = img.size
    for y in range(h):
        for x in range(w):
            if pixdata[x, y] < threshold:
                pixdata[x, y] = 0
            else:
                pixdata[x, y] = 255
    return img

#把图像转为灰度图
img = Image.open('/Users/wywy/Desktop/MNIST/40_3.jpg').convert("L")

#领域像素算法
def depoint(img):   #input: gray image
    pixdata = img.load()
    w,h = img.size
    for y in range(1,h-1):
        for x in range(1,w-1):
            count = 0
            if pixdata[x,y-1] < 245:
                count = count + 1
            if pixdata[x,y+1] < 245:
                count = count + 1
            if pixdata[x-1,y] < 245:
                count = count + 1
            if pixdata[x+1,y] < 245:
                count = count + 1
            if count < 2:
                pixdata[x,y] = 255
    return img

img= depoint(img)
img.save('/Users/wywy/Desktop/xxx/40_3.jpg')
