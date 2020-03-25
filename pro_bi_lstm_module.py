# [1]
# import the library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn import metrics
import os

import torch
import torch.nn as nn
import torch.utils.data as Data # Abstract class for data set in pytorch
import torchvision
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch import optim


# [2]
# load the data
PATH='F:/Tinky/大学课件及作业/5 二三四课/5-7.科研训练/陈熹/new dataset/'

r01=pd.read_excel(PATH+'r_01.xlsx')
r02=pd.read_excel(PATH+'r_02.xlsx')
s01=pd.read_excel(PATH+'s_01.xlsx')
s02=pd.read_excel(PATH+'s_02.xlsx')
t01=pd.read_excel(PATH+'t_01.xlsx')
w01=pd.read_excel(PATH+'w_01.xlsx')


# [3]
# ignore useless columns
r01=r01.loc[:,'ax(g)':'AngleZ(deg)']
r02=r02.loc[:,'ax(g)':'AngleZ(deg)']
s01=s01.loc[:,'ax(g)':'AngleZ(deg)']
s02=s02.loc[:,'ax(g)':'AngleZ(deg)']
t01=t01.loc[:,'ax(g)':'AngleZ(deg)']
w01=w01.loc[:,'ax(g)':'AngleZ(deg)']

# check the data
print("r:",r01.shape,r02.shape)
print("s:",s01.shape,s02.shape)
print("t:",t01.shape)
print("w:",w01.shape)


# [4]
# Scale the data
from sklearn.preprocessing import MaxAbsScaler
MAS=MaxAbsScaler()

def MinMaxScale(df):
    col1 = pd.DataFrame(MAS.fit_transform(df.iloc[:, 0:3]))
    col2 = pd.DataFrame(MAS.fit_transform(df.iloc[:, 3:6]))
    col3 = pd.DataFrame(MAS.fit_transform(df.iloc[:, 6:9]))
    df = pd.DataFrame(pd.concat([col1, col2, col3], axis=1))
    return df

r01=MinMaxScale(r01)
r02=MinMaxScale(r02)
s01=MinMaxScale(s01)
s02=MinMaxScale(s02)
t01=MinMaxScale(t01)
w01=MinMaxScale(w01)


# [5]
# Sec to pic
ws = 50  # window size
ss = 25  # step size
'''
orderly split the data

def window_split(df):
    n=(len(df)-ws)//ss+1    # n windows
    all = []
    for i in range(n):
        ser = np.array(df[ss*i:ss*i+ws])
        all.append(ser)
    return all
'''


'''
randomly split the data
'''
import random
def window_split(df):
    n=(len(df)-ws)//ss+1    # n windows
    all = []
    for i in range(n):
        rdm=random.randint(0, ss*(n-1)) # choose a window_split start randomly
        ser = np.array(df[rdm:rdm+ws])
        all.append(ser)
    return all



r01=np.array(window_split(r01))
r02=np.array(window_split(r02))
s01=np.array(window_split(s01))
s02=np.array(window_split(s02))
t01=np.array(window_split(t01))
w01=np.array(window_split(w01))


# [6]
# combine the same type
R=np.concatenate((r01,r02),axis=0)
S=np.concatenate((s01,s02),axis=0)
T=t01
W=w01
# check the shape
print(R.shape)  # should be (x,ws,9)
print(S.shape)
print(T.shape)
print(W.shape)
# construct the train set
X=np.concatenate((R,S,T,W),axis=0)
# label the data
Y=np.array([0]*R.shape[0]+[1]*S.shape[0]+[2]*T.shape[0]+[3]*W.shape[0])
print(X.shape,Y.shape)  # check the shape


# [7]
# split into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size=0.33, random_state=42)

# flatten (no flatten when using Pytorch)
# X_train.shape=(len(X_train),ws*9)
# X_test.shape=(len(X_test),ws*9)
'''
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(y_test.shape)
type now: numpy.ndarray
X_train.shape: (x, 50, 9)
Y_train.shape: (x,)
X_test.shape: (y, 50, 9)
Y_test.shape: (y,)
'''

# transform to pytorch tensor
Xt_train=torch.from_numpy(X_train)
Yt_train=torch.from_numpy(Y_train)
Xt_test=torch.from_numpy(X_test)
Yt_test=Y_test  # do not need to transform
'''
Xt_train/Yt_train/Xt_test: torch.tensor
Yt_test: numpy.ndarray
'''
print(Xt_train.shape)
print(Yt_train.shape)
print(Xt_test.shape)
print(Yt_test.shape)
'''
output:
torch.Size([x, 9, 50])
torch.Size([x])
torch.Size([y, 9, 50])
(y,)
'''

# [8]
# LSTM Model
#parameters
epochs = 8      # training times(all the data)
batch_size = 1  # updated the parameter after receiving one picture
time_step = 50  # time step/image height
input_size = 9 # input_size / image width
lr = 0.01

class Rnn(nn.Module):
    def __init__(self):
        super(Rnn,self).__init__()

        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=18,
            num_layers=1,
            batch_first=True,  #（time_step,batch,input）时是Ture
        )
        self.out = nn.Linear(18,10)

    def forward(self, x):
        r_out,(h_n,h_c) = self.rnn(x,None)  # x (batch,time_step,input_size)
        out = self.out(r_out[:,-1,:]) #(batch,time_step,input)
        return out

rnn = Rnn()
print(rnn)

optimizer = optim.Adam(rnn.parameters(),lr=lr)
loss_func = nn.CrossEntropyLoss()

for epoch in range(epochs):
    for step in range(len(Xt_train)//batch_size):
        b_x = Xt_train[step*batch_size:(step+1)*(batch_size)] # b_x torch.Size([batch_size,9,50])
        b_y = Yt_train[step*batch_size:(step+1)*(batch_size)] # b_y torch.Size([batch_size])
        output = rnn(b_x.float())  # output torch.Size([batch_size,10])
        loss = loss_func(output, b_y.long())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 20 == 0:
            test_out = rnn(Xt_test.float())  # (sample,time_step,input_size)
            pred_y = torch.max(test_out, 1)[1].data.numpy().squeeze()
            accuracy = sum(pred_y == Yt_test) / Yt_test.size
            print('Epoch:', epoch, '|train loss:' + str(loss.item()), '|test accuracy:' + str(accuracy))

# print 10 prediction from test data
test_output = rnn(Xt_test[:10].float())
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()  # pred_y numpy.ndarray (10,)
print(pred_y, 'prediction number')
print(Yt_test[:10], 'real number')


