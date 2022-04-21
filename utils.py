import numpy as np
import torch
from sklearn.decomposition import PCA



def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.01)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.01)
        m.bias.data.fill_(0)

def kappa(testData, k):
    dataMat = np.mat(testData)
    P0 = 0.0
    for i in range(k):
        P0 += dataMat[i, i] * 1.0
    xsum = np.sum(dataMat, axis=1)
    ysum = np.sum(dataMat, axis=0)
    Pe = float(ysum * xsum) / np.sum(dataMat) ** 2
    P0 = float(P0 / np.sum(dataMat) * 1.0)
    cohens_coefficient = float((P0 - Pe) / (1 - Pe))
    return cohens_coefficient

def test(predict, labels):
    correct = 0
    pred = predict.data.max(1)[1]
    correct = pred.eq(labels.data).cpu().sum()
    return correct, len(labels.data)


# 在周围填充零值
def flip(data):  # data:  145, 145, 3

    y_4 = np.zeros_like(data)
    y_1 = y_4
    y_2 = y_4
    first = np.concatenate((y_1, y_2, y_1), axis=1)
    second = np.concatenate((y_4, data, y_4), axis=1)
    third = first
    Data = np.concatenate((first, second, third), axis=0)
    return Data

def splitTrainTestSet(X,y,testRatio,randomState=345):
    X_train,X_test,y_train,y_test=X*(1-testRatio),X*testRatio,y*(1-testRatio),y*testRatio
    return X_train,X_test,y_train,y_test

#  加padding
def padWithZeros(X,margin=2):
    newX=np.zeros((X.shape[0]+2*margin,X.shape[1]+2*margin,X.shape[2]))
    newX[margin:X.shape[0]+margin,margin:X.shape[1]+margin,:] = X
    return newX


def createImageCubes(X,y,windowSize=5,removeZeroLabels=True):
    margin=int((windowSize-1)/2)
    zeroPaddedX=padWithZeros(X,margin=margin)
    patchesData=np.zeros((X.shape[0]*X.shape[1],windowSize,windowSize,X.shape[2]))
    patchesLabels=np.zeros((X.shape[0]*X.shape[1]))
    patchIndex=0
    for r in range(margin,zeroPaddedX.shape[0]-margin):
        for c in range(margin,zeroPaddedX.shape[1]-margin):
            patch=zeroPaddedX[r-margin:r+margin+1,c-margin:c+margin+1]
            patchesData[patchIndex,:,:,:]=patch
            patchesLabels[patchIndex]=y[r-margin,c-margin]
            patchIndex=patchIndex+1
    if removeZeroLabels:
        patchesData=patchesData[patchesLabels>0,:,:,:]
        patchesLabels=patchesLabels[patchesLabels>0]
        patchesLabels-=1
    return patchesData,patchesLabels

#  PCA变换
def applyPCA(X,numComponents):
    newX=np.reshape(X,(-1,X.shape[2]))
    pca=PCA(n_components=numComponents,whiten=True)
    newX=pca.fit_transform(newX)
    newX=np.reshape(newX,(X.shape[0],X.shape[1],numComponents))
    return newX





