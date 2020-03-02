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
def window_split(df):
    n=(len(df)-ws)//ss+1
    all = []
    for i in range(n):
        ser = np.array(df[ss*i:ss*i+ws])
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
y=np.array([0]*R.shape[0]+[1]*S.shape[0]+[2]*T.shape[0]+[3]*W.shape[0])
print(X.shape,y.shape)  # check the shape

# split into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=42)

# flatten
X_train.shape=(len(X_train),ws*9)
X_test.shape=(len(X_test),ws*9)


# [7]
# try some basic models
import sklearn.metrics as metric

# kNN models
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train, y_train)
predVal = clf.predict(X_test)
actVal = y_test
print(metric.confusion_matrix(actVal, predVal))
print(metric.accuracy_score(actVal, predVal))

# logistic regression
from sklearn.linear_model import LogisticRegression
log = LogisticRegression()
log.fit(X_train, y_train)
predVal = log.predict(X_test)
actVal = y_test
print(metric.confusion_matrix(actVal, predVal))
print(metric.accuracy_score(actVal, predVal))

# Decision Tree
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(max_depth=8,random_state=0)
tree.fit(X_train, y_train)
predVal = tree.predict(X_test)
actVal = y_test
print(metric.confusion_matrix(actVal, predVal))
print(metric.accuracy_score(actVal, predVal))

# GBDT
from sklearn.ensemble import GradientBoostingClassifier
gbrt = GradientBoostingClassifier(random_state=0)
gbrt.fit(X_train, y_train)
predVal = gbrt.predict(X_test)
actVal = y_test
print(metric.confusion_matrix(actVal, predVal))
print(metric.accuracy_score(actVal, predVal))

# MLP
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(solver='lbfgs', random_state=0)
mlp.fit(X_train, y_train)
predVal = mlp.predict(X_test)
actVal = y_test
print(metric.confusion_matrix(actVal, predVal))
print(metric.accuracy_score(actVal, predVal))

# SVM models
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline

pca = PCA(n_components=150,whiten=True, random_state=42)
svc = SVC(kernel='rbf',class_weight='balanced')
model = make_pipeline(pca, svc)

from sklearn.model_selection import GridSearchCV
param_grid = {'svc__C': [0.1, 0.5, 1, 5, 10],
              'svc__gamma': [0.0001, 0.0005, 0.001, 0.05, 0.1]}
grid = GridSearchCV(model, param_grid)
grid.fit(X_train, y_train)
print(grid.best_params_)
model = grid.best_estimator_
y_fit = model.predict(X_test)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_fit))


