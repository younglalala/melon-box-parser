import numpy as np

aa=[0,2]
def onehott(info):
    # xx = 'ABCDEFGX'
    num = [0, 1, 2, 3, 4, 5, 6,7]
    all_onehot=[]
    for n in range(len(num)):
        one_hot=[]
        for j in range(len(num)):
            one_hot.append(0)
        one_hot[num[n]]=1
        all_onehot.append(one_hot)
    dict_infor=dict(zip(num,all_onehot))
    input_infor=[]
    info.sort()

    new_input=[]
    for b in range(7):
        new_input.append(7)
    for inf in info:
        if len(info)==0:
            new_input.append(7)
        else:
            new_input[inf]=inf

    for numberr in new_input:
        input_infor.append(dict_infor.get(numberr))

    return np.array(input_infor)

print(onehott(aa))












