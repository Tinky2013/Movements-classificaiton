# [1]
# import the library
import numpy as np
import pandas as pd
import matplotlib
# import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn import metrics
import os

import torch
import torch.nn as nn
import torch.utils.data as Data # Abstract class for data set in pytorch
import torchvision
from torch.autograd import Variable


# [2]
# load the data

PATH='F:/Tinky/大学课件及作业/5 二三四课/5-7.科研训练/陈熹/dataset/'

# flat movement
fa=open(PATH+'fl_move_train.csv')
fb=open(PATH+'fl_move_test.csv')
da1=pd.read_csv(fa)
db1=pd.read_csv(fb)

# flat rotation
fa=open(PATH+'fl_rott_train.csv')
fb=open(PATH+'fl_rott_test.csv')
da2=pd.read_csv(fa)
db2=pd.read_csv(fb)

# vertical movement
fa=open(PATH+'up_move_train.csv')
fb=open(PATH+'up_move_test.csv')
da3=pd.read_csv(fa)
db3=pd.read_csv(fb)

# vertical rotation
fa=open(PATH+'up_move_train.csv')
fb=open(PATH+'up_move_test.csv')
da4=pd.read_csv(fa)
db4=pd.read_csv(fb)

# [3]
# ignore useless columns
da1=da1.loc[:,'ax(g)':'AngleZ(deg)']
db1=db1.loc[:,'ax(g)':'AngleZ(deg)']
da2=da2.loc[:,'ax(g)':'AngleZ(deg)']
db2=db2.loc[:,'ax(g)':'AngleZ(deg)']
da3=da3.loc[:,'ax(g)':'AngleZ(deg)']
db3=db3.loc[:,'ax(g)':'AngleZ(deg)']
da4=da4.loc[:,'ax(g)':'AngleZ(deg)']
db4=db4.loc[:,'ax(g)':'AngleZ(deg)']
# check the data
print("class 1:",da1.shape,db1.shape)
print("class 2:",da2.shape,db2.shape)
print("class 3:",da3.shape,db3.shape)
print("class 4:",da4.shape,db4.shape)

# [4]
# normalization
def Norm(df):
    df1 = df.loc[:, 'ax(g)':'az(g)']
    df1 = (df1 - df1.mean()) / (df1.std())

    df2 = df.loc[:, 'wx(deg/s)':'wz(deg/s)']
    df2 = (df2 - df2.mean()) / (df2.std())

    df3 = df.loc[:, 'AngleX(deg)':'AngleZ(deg)']
    df3 = (df3 - df3.mean()) / (df3.std())

    df = pd.concat([df1, df2, df3], axis=1, ignore_index=True)
    return df

da1 = Norm(da1)
db1 = Norm(db1)
da2 = Norm(da2)
db2 = Norm(db2)
da3 = Norm(da3)
db3 = Norm(db3)
da4 = Norm(da4)
db4 = Norm(db4)


# [5]
# split the data
def window_split(data):
    '''
    data: dimension=2
    alpha: range of coincidence
    series_len must be integer!
    '''
    window_size = 128
    alpha = 0.5
    series_len = int(window_size * alpha)
    step = int((data.shape[0]-window_size*(1-alpha)) / series_len)
    all = []
    for i in range(step):
        ser = np.array(data[series_len*i:series_len*i+window_size])
        all.append(ser)
    return all

da1=np.array(window_split(da1))
db1=np.array(window_split(db1))
da2=np.array(window_split(da2))
db2=np.array(window_split(db2))
da3=np.array(window_split(da3))
db3=np.array(window_split(db3))
da4=np.array(window_split(da4))
db4=np.array(window_split(db4))

# check the dimension
print(da1.shape,db1.shape)
print(da2.shape,db2.shape)
print(da3.shape,db3.shape)
print(da4.shape,db4.shape)


# [6]
# transform to torch tensor
# recopy the data here
Da1=torch.from_numpy(da1)
Db1=torch.from_numpy(db1)
Da2=torch.from_numpy(da2)
Db2=torch.from_numpy(db2)
Da3=torch.from_numpy(da3)
Db3=torch.from_numpy(db3)
Da4=torch.from_numpy(da4)
Db4=torch.from_numpy(db4)

print(Da1.shape,Db1.shape)
print(Da2.shape,Db2.shape)
print(Da3.shape,Db3.shape)
print(Da4.shape,Db4.shape)

# [7] create the label
train_lb=np.array([0]*19+[1]*18+[2]*16+[3]*16)
test_lb=np.array([0]*10+[1]*7+[2]*4+[3]*4)
ya=torch.from_numpy(train_lb)
yb=torch.from_numpy(test_lb)
print(ya.shape,yb.shape)


# [8]
# CNN
# transform the data
Da1 = Variable(torch.unsqueeze(Da1, dim=1))
Db1 = Variable(torch.unsqueeze(Db1, dim=1))
Da2 = Variable(torch.unsqueeze(Da2, dim=1))
Db2 = Variable(torch.unsqueeze(Db2, dim=1))
Da3 = Variable(torch.unsqueeze(Da3, dim=1))
Db3 = Variable(torch.unsqueeze(Db3, dim=1))
Da4 = Variable(torch.unsqueeze(Da4, dim=1))
Db4 = Variable(torch.unsqueeze(Db4, dim=1))

# concat
xa=torch.cat((Da1,Da2,Da3,Da4),0)
xb=torch.cat((Db1,Db2,Db3,Db4),0)
print(xa.shape,xb.shape)

# Hyper parameters
EPOCH=4     # how many times we train the whole data
BATCH_SIZE=5 # train 5 each
LR=0.001    # learning rate
torch.manual_seed(1)

# structure
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         # input shape (1, 128, 9)
            nn.Conv2d(
                in_channels=1,              # input height
                out_channels=16,            # n_filters
                kernel_size=3,              # filter size
                stride=1,                   # filter movement/step
                padding=1,                  # if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (16, 128, 9)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(kernel_size=1),    # choose max value in 2x2 area, output shape (16, 128, 9)
        )
        self.conv2 = nn.Sequential(         # input shape (16, 128, 9)
            nn.Conv2d(16, 32, 3, 1, 1),     # output shape (32, 128, 9)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(1),                # output shape (32, 128, 9)
        )
        self.out = nn.Linear(32 * 128 * 9, 4)   # fully connected layer, output 4 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)           # flatten the output of conv2 to (batch_size, 32 * 128 * 9)
        output = self.out(x)
        return output, x    # return x for visualization

cnn=CNN()
print(cnn)

# [9]
# training
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()      # the target label is not one-hottedfor epoch in range(EPOCH):
for epoch in range(EPOCH):
    for i in range(int(xa.shape[0]/BATCH_SIZE)):
        # b_x, b_y: torch.tensor
        b_x = Variable(xa[i*BATCH_SIZE:(i+1)*BATCH_SIZE])  # torch.Size([BATCH_SIZE, 1, 28, 28])
        b_y = Variable(ya[i*BATCH_SIZE:(i+1)*BATCH_SIZE])  # torch.Size([BATCH_SIZE])
        b_x = torch.tensor(b_x, dtype=torch.float32)
        b_y = torch.tensor(b_y, dtype=torch.int64)


        output = cnn(b_x)[0]               # cnn output
        loss = loss_func(output, b_y)   # cross entropy loss
        optimizer.zero_grad()           # clear gradients for this training step
        loss.backward()                 # backpropagation, compute gradients
        optimizer.step()                # apply gradients

        # print the training process: Epoch, loss and accuracy
        # if i % 10 == 0:
        xb = torch.tensor(xb, dtype=torch.float32)
        yb = torch.tensor(yb, dtype=torch.int64)
        test_output, last_layer = cnn(xb)
        pred_y = torch.max(test_output, 1)[1].data.squeeze()
        accuracy = (pred_y == yb).sum().item() / float(yb.size(0))
        print('Epoch: ', epoch, '| train loss: %.4f' % loss.item(), '| test accuracy: %.2f' % accuracy)

# We can check the performance with 10 new data
# print 10 predictions from test data
xb = torch.tensor(xb, dtype=torch.float32)
test_output, _ = cnn(xb[:20])
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
print(pred_y, 'prediction number')
print(yb.numpy()[:20], 'real number')
