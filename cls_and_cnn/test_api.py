from requests_toolbelt import MultipartEncoder
import requests
import os, sys


dir="/Users/wywy/Desktop/text_image/"
dir1="/Users/wywy/Desktop/json/"
b=0
for ii in os.listdir(dir):
    b+=1
    m = MultipartEncoder(
        fields={
            'APPID':'1256752352',
            'api_key': '84mHZfjcIl26yd6LNDpEQPBl7GEyUnuo',
            'api_secret': 'AKIDazBByYh63nKgplG1LSL1TmskQs8BEe4H',
            # 'image_file': ('filename', open('./timg.jpg', 'rb'), 'image/png')
            'image_file': ('filename', open(str(dir+str(ii)), 'rb'), 'image/png')
        })

    r = requests.post('http://recognition.image.myqcloud.com/ocr/handwriting', data=m,
                      headers={'Content-Type': m.content_type})

    d=r.content
    # print(r.content,'11')
    # print(type(r),'22')
    # print(type(r.json()),'33')
    # print(r.json(),'44')
    a=r.json()
# 创建空文件
    arr=ii.split(".")
    text=str(arr[0]+".json")

    text1=dir1+text
    print(text1)
    # if os.path.exists(text1):
    #     #     # 删除文件，可使用以下两种方法。
    #     #     os.remove(text1)
    #     #     # os.unlink(my_file)
    #     # else:
    #     #     os.mknod(text1)
    fd = os.open(text1,os.O_RDWR|os.O_CREAT)
    os.write(fd,bytes(str(a), 'UTF-8'))
    print("第{0}次：{1}文件保存成功".format(b,text))


#
