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
# plot the data
rdf=pd.DataFrame()
for col in r01.columns:
    rdf[col]=r01[col]
rdf[:800].plot(subplots=True,legend=False)
plt.show()

sdf=pd.DataFrame()
for col in s01.columns:
    sdf[col]=s01[col]
sdf[:800].plot(subplots=True,legend=False)
plt.show()

tdf=pd.DataFrame()
for col in t01.columns:
    tdf[col]=t01[col]
tdf[:800].plot(subplots=True,legend=False)
plt.show()

wdf=pd.DataFrame()
for col in w01.columns:
    wdf[col]=w01[col]
wdf[:800].plot(subplots=True,legend=False)
plt.show()

# [5]
# scale and plot
from sklearn.preprocessing import MaxAbsScaler
MAS=MaxAbsScaler()

def MinMaxScale(k):
    col1 = pd.DataFrame(MAS.fit_transform(k.iloc[:, 0:3]))
    col2 = pd.DataFrame(MAS.fit_transform(k.iloc[:, 3:6]))
    col3 = pd.DataFrame(MAS.fit_transform(k.iloc[:, 6:9]))
    k = pd.DataFrame(pd.concat([col1, col2, col3], axis=1))
    return k

rdf_scale=MinMaxScale(rdf)
plt.matshow(rdf_scale[:800].T, interpolation=None, aspect='auto')
plt.show()

sdf_scale=MinMaxScale(sdf)
plt.matshow(sdf_scale[:800].T, interpolation=None, aspect='auto')
plt.show()

tdf_scale=MinMaxScale(tdf)
plt.matshow(tdf_scale[:800].T, interpolation=None, aspect='auto')
plt.show()

wdf_scale=MinMaxScale(wdf)
plt.matshow(wdf_scale[:800].T, interpolation=None, aspect='auto')
plt.show()