import numpy as np

# data=[3,2,5,10,8,6]
def one_hot(data):
    max_x=max(data)
    data_onehot=[]
    for i in data:
        data_onehot1=[]
        for j in range(max_x+1):
            data_onehot1.append(0)
        data_onehot1[i]=1
        data_onehot.append(data_onehot1)

    return np.array(data_onehot)


