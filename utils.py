import numpy as np
import torch
import os
import scipy.io as sio
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

def getData(dataset_name, folder):
    # load data
    if dataset_name == 'Indian':
        # path_data = folder + '/IndianPines/' + 'Indian_pines_corrected.mat'
        path_data = os.path.join(folder, os.path.join('IndianPines', 'Indian_pines_corrected.mat'))
        # path_gt = folder + '/' + 'Indian_pines_gt.mat'
        path_gt = os.path.join(folder, os.path.join('IndianPines', 'Indian_pines_gt.mat'))

        X = sio.loadmat(path_data)['indian_pines_corrected']
        y = sio.loadmat(path_gt)['indian_pines_gt']
    elif dataset_name == 'Botswana':
        # path_data = folder + '/' + 'Botswana.mat'
        path_data = os.path.join(folder, os.path.join('Botswana', 'Botswana.mat'))

        # path_gt = folder + '/' + 'Botswana_gt.mat'
        path_gt = os.path.join(folder, os.path.join('Botswana', 'Botswana_gt.mat'))

        X = sio.loadmat(path_data)['Botswana']
        y = sio.loadmat(path_gt)['Botswana_gt']
    elif dataset_name == 'PaviaC':
        # path_data = folder + '/' + 'Pavia.mat'
        path_data = os.path.join(folder, os.path.join('PaviaC', 'Pavia.mat'))

        # path_gt = folder + '/' + 'Pavia_gt.mat'
        path_gt = os.path.join(folder, os.path.join('PaviaC', 'Pavia_gt.mat'))

        X = sio.loadmat(path_data)['pavia']
        y = sio.loadmat(path_gt)['pavia_gt']
    elif dataset_name == 'PaviaU':
        # path_data = folder + '/' + 'Pavia.mat'
        path_data = os.path.join(folder, os.path.join('PaviaU', 'PaviaU.mat'))

        # path_gt = folder + '/' + 'Pavia_gt.mat'
        path_gt = os.path.join(folder, os.path.join('PaviaU', 'PaviaU_gt.mat'))

        X = sio.loadmat(path_data)['paviaU']
        y = sio.loadmat(path_gt)['paviaU_gt']
    elif dataset_name == 'Salinas':
        # path_data = folder + '/' + 'Pavia.mat'
        path_data = os.path.join(folder, os.path.join('salinas', 'Salinas_corrected.mat'))

        # path_gt = folder + '/' + 'Pavia_gt.mat'
        path_gt = os.path.join(folder, os.path.join('salinas', 'Salinas_gt.mat'))

        X = sio.loadmat(path_data)['salinas_corrected']
        y = sio.loadmat(path_gt)['salinas_gt']
    elif dataset_name == 'KSC':
        # path_data = folder + '/' + 'Pavia.mat'
        path_data = os.path.join(folder, os.path.join('KSC', 'KSC.mat'))

        # path_gt = folder + '/' + 'Pavia_gt.mat'
        path_gt = os.path.join(folder, os.path.join('KSC', 'KSC_gt.mat'))

        X = sio.loadmat(path_data)['KSC']
        y = sio.loadmat(path_gt)['KSC_gt']
    elif dataset_name == 'yumi':
        # path_data = folder + '/' + 'yumidata_new.mat'
        path_data = os.path.join(folder, os.path.join('yumi', 'yumidata_new.mat'))
        # path_gt = folder + '/' + 'yumilabel_new2.mat'
        path_gt = os.path.join(folder, os.path.join('yumi', 'yumilabel_new2.mat'))

        X = sio.loadmat(path_data)['yumidata']
        y = sio.loadmat(path_gt)['yumi_label']
    
    return X, y






