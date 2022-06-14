from __future__ import print_function
import argparse
import imp
import os
import random
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from utils import applyPCA, kappa, test, flip, padWithZeros, createImageCubes, splitTrainTestSet
from datasets import TrainDS, TestDS
# from model import netD, netG
from model import netG
from transformer import ADGANTransformer as netD
from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--imageSize', type=int, default=28, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=30, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--decreasing_lr', default='10,20,30,40,50,60,80', help='decreasing strategy')
parser.add_argument('--wd', type=float, default=0.001, help='weight decay')
parser.add_argument('--clamp_lower', type=float, default=-0.01)
parser.add_argument('--clamp_upper', type=float, default=0.01)

parser.add_argument('--dataset', default='Indian', help="Dataset to use.")
parser.add_argument('--folder', default='./data', help="Folder where to store the datasets")
parser.add_argument('--nTrain', type=int, default=2000, help='how many data to train')

opt = parser.parse_args()
opt.outf = 'model'
# opt.cuda = False
print(opt)
folder = opt.folder
dataset_name = opt.dataset
nTrain = opt.nTrain

CRITIC_ITERS = 1
train_generator = True
try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)
cudnn.benchmark = False

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# num_class = 16
# load data
# matfn1 = 'D:\RocZhang\data\IndianPines\Indian_pines_corrected.mat'
# data1 = sio.loadmat(matfn1)
# X = data1['indian_pines_corrected']
# matfn2 = 'D:\RocZhang\data\IndianPines\Indian_pines_gt.mat'
# data2 = sio.loadmat(matfn2)
# y = data2['indian_pines_gt']

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
elif dataset_name == 'yumi':
    # path_data = folder + '/' + 'yumidata_new.mat'
    path_data = os.path.join(folder, os.path.join('yumi', 'yumidata_new.mat'))
    # path_gt = folder + '/' + 'yumilabel_new2.mat'
    path_gt = os.path.join(folder, os.path.join('yumi', 'yumilabel_new2.mat'))

    X = sio.loadmat(path_data)['yumidata']
    y = sio.loadmat(path_gt)['yumi_label']


# test_ratio = 0.90
# patch_size = 25
pca_components = 3
print('Hyperspectral data shape:', X.shape)
print('Label shape:', y.shape)
X_pca = applyPCA(X, numComponents=pca_components)
print('Data shape after PCA :', X_pca.shape)

[nRow, nColumn, nBand] = X_pca.shape  # nColum 145 nBand 3 nRow 145
pcdata = flip(X_pca)  # 435 435 3
groundtruth = flip(y)  # 435 435

num_class = int(np.max(y))

HalfWidth = 32
# Wid = 2 * HalfWidth
# G = groundtruth[145 - 32: 2 * 145 + 32, 145 - 32: 2* 145 + 32] = 209, 209
# data = pcdata[145 - 32: 2 * 145 + 32, 145 - 32: 2* 145 + 32] = 209, 209, 3
G = groundtruth[nRow - HalfWidth:2 * nRow + HalfWidth, nColumn - HalfWidth:2 * nColumn + HalfWidth]
data = pcdata[nRow - HalfWidth:2 * nRow + HalfWidth, nColumn - HalfWidth:2 * nColumn + HalfWidth, :]
# row, col  = 209, 209
[row, col] = G.shape

NotZeroMask = np.zeros([row, col])  # 创建了一个209，209的二维向量，值全部为零
# Wid = 2 * HalfWidth
# NotZeroMask[32 + 1: -1 - 32 + 1, 32 + 1: -1 - 32 + 1] = NotZeroMask[33: -32, 33, -32] = 1
# 就是把选定区域的值全部变成了1
NotZeroMask[HalfWidth + 1: -1 - HalfWidth + 1, HalfWidth + 1: -1 - HalfWidth + 1] = 1
# 上面选定区域的值不变，其他值变为零
G = G * NotZeroMask
# 取出值不为0的坐标，row是横坐标，column是纵坐标
[Row, Column] = np.nonzero(G)
nSample = np.size(Row)

RandPerm = np.random.permutation(nSample)  # 洗牌，返回随机值


# nTrain = 5000
nTest = nSample - nTrain
imdb = {}
imdb['datas'] = np.zeros([2 * HalfWidth, 2 * HalfWidth, nBand, nTrain + nTest], dtype=np.float32)  # 64， 64，3，10176
imdb['Labels'] = np.zeros([nTrain + nTest], dtype=np.int64)  # 10176
imdb['set'] = np.zeros([nTrain + nTest], dtype=np.int64)  # 10176
# data[Row[]]
#
for iSample in range(nTrain + nTest):  # 将训练集随机取值放进imdb中
    # print('Row[RandPerm[iSample]] - HalfWidth: Row[RandPerm[iSample]] + HalfWidth = {}: {}'.format(Row[RandPerm[iSample]] - HalfWidth, Row[RandPerm[iSample]] + HalfWidth))
    # print('Column[RandPerm[iSample]] - HalfWidth: Column[RandPerm[iSample]] + HalfWidth = {}: {}'.format(Column[RandPerm[iSample]] - HalfWidth, Column[RandPerm[iSample]] + HalfWidth))
    imdb['datas'][:, :, :, iSample] = data[Row[RandPerm[iSample]] - HalfWidth: Row[RandPerm[iSample]] + HalfWidth,
                                      Column[RandPerm[iSample]] - HalfWidth: Column[RandPerm[iSample]] + HalfWidth,
                                      :]
    # print('Row[RandPerm[iSample]],Column[RandPerm[iSample]] = {}, {}'.format(Row[RandPerm[iSample]], Column[RandPerm[iSample]]))
    imdb['Labels'][iSample] = G[Row[RandPerm[iSample]],
                                Column[RandPerm[iSample]]].astype(np.int64)
print('Data is OK.')

imdb['Labels'] = imdb['Labels'] - 1

imdb['set'] = np.hstack((np.ones([nTrain]), 3 * np.ones([nTest]))).astype(np.int64)
Xtrain = imdb['datas'][:, :, :, :nTrain]  # 取两千个作为训练集
ytrain = imdb['Labels'][:nTrain]  # 取两千个作为训练集
print('Xtrain :', Xtrain.shape)
print('yTrain:', ytrain.shape)
result = Counter(ytrain)
print(result)

Xtest = imdb['datas']  # 所有的数据作为测试集
ytest = imdb['Labels']  # 所有的数据作为测试集
print('Xtest :', Xtest.shape)
print('ytest:', ytest.shape)
"""
Xtrain=Xtrain.reshape(-1,patch_size,patch_size,pca_components)
Xtest=Xtest.reshape(-1,patch_size,patch_size,pca_components)
print(' before Xtrain shape:',Xtrain.shape)
print('before Xtest shape:',Xtest.shape)
"""
# 将数据转化成dataset
Xtrain = Xtrain.transpose(3, 2, 0, 1)
Xtest = Xtest.transpose(3, 2, 0, 1)
print('after Xtrain shape:', Xtrain.shape)
print('after Xtest shape:', Xtest.shape)




# 创建 trainloader 和 testloader
trainset = TrainDS(Xtrain, ytrain)
testset = TestDS(Xtest, ytest)
train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=200, shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=200, shuffle=False, num_workers=0)

nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)

nc = pca_components
nb_label = num_class
print("label", nb_label)

netG = netG(nz, ngf, nc)

if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

netD = netD(img_size=64, in_chans=nc, num_classes=nb_label + 1, window_size=8, patch_size=22)

if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

s_criterion = nn.BCELoss()
c_criterion = nn.NLLLoss()

input = torch.FloatTensor(opt.batchSize, nc, opt.imageSize, opt.imageSize)
noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
fixed_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1)
s_label = torch.FloatTensor(opt.batchSize)
c_label = torch.LongTensor(opt.batchSize)
f_label = torch.LongTensor(opt.batchSize)

real_label = 0.8
fake_label = 0.2

if opt.cuda:
    netD.cuda()
    netG.cuda()
    s_criterion.cuda()
    c_criterion.cuda()
    input, s_label = input.cuda(), s_label.cuda()
    c_label = c_label.cuda()
    f_label = f_label.cuda()
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

input = Variable(input)
s_label = Variable(s_label)
c_label = Variable(c_label)
f_label = Variable(f_label)
noise = Variable(noise)
fixed_noise = Variable(fixed_noise)

optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(0.9, 0.999), weight_decay=0.02)
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(0.9, 0.999), weight_decay=0.005)

decreasing_lr = list(map(int, opt.decreasing_lr.split(',')))
print('decreasing_lr: ' + str(decreasing_lr))


best_acc = 0

for epoch in range(1, opt.niter + 1):
    netD.train()
    netG.train()
    right = 0
    if epoch in decreasing_lr:
        optimizerD.param_groups[0]['lr'] *= 0.9
        optimizerG.param_groups[0]['lr'] *= 0.9

    for i, datas in enumerate(train_loader):
        for j in range(10):  ## Update D 10 times for every G epoch
            netD.zero_grad()

            # 将真实的训练数据放入到判别器网络中，得到判别器输出的标签
            img, label = datas
            batch_size = img.size(0)
            input.resize_(img.size()).copy_(img)
            s_label.resize_(batch_size).fill_(real_label)
            c_label.resize_(batch_size).copy_(label)
            # 这里很奇怪，输出的值都是负值
            c_output = netD(input)  # 200，17

            # s_errD_real = s_criterion(s_output, s_label)
            # 通过判别器输出的标签和真实标签，得到真实数据的损失
            c_errD_real = c_criterion(c_output, c_label)
            errD_real = c_errD_real
            # 反向传播优化函数
            errD_real.backward()
            # 计算出所有标签的平均值
            D_x = c_output.data.mean()

            correct, length = test(c_output, c_label)
            # print('real train finished!')

            # train with fake
            # 生成噪声
            noise.resize_(batch_size, nz, 1, 1)
            noise.normal_(0, 1)
            noise_ = np.random.normal(0, 1, (batch_size, nz, 1, 1))
            with torch.no_grad():
                noise.resize_(batch_size, nz, 1, 1).copy_(torch.from_numpy(noise_))

            # label = np.random.randint(0, nb_label, batch_size)
            # 生成标签
            label = np.full(batch_size, nb_label)
            with torch.no_grad():
                # f_label.data.resize_(batch_size).copy_(torch.from_numpy(label))
                # 形状为（200，），值为16的标签
                f_label.resize_(batch_size).copy_(torch.from_numpy(label))

            fake = netG(noise)
            # s_label.fill_(fake_label)
            c_output = netD(fake.detach())
            # s_errD_fake = s_criterion(s_output, s_label)
            # 单独训练噪声类，高光谱图像有16类，分别对应1-15，噪声类分配为16
            c_errD_fake = c_criterion(c_output, f_label)
            errD_fake = c_errD_fake
            # 反向传播
            errD_fake.backward()
            # 计算出生成图像类别的平均值
            D_G_z1 = c_output.data.mean()
            # 判别器的损失是真实图像的损失加上生成图像的损失
            errD = errD_real + errD_fake
            # 优化
            optimizerD.step()
            # print('fake train finished!')
            ###############
            #  Updata G
            ##############

        netG.zero_grad()
        # s_label.data.fill_(real_label)  # fake labels are real for generator cost
        c_output = netD(fake)
        # s_errG = s_criterion(s_output, s_label)
        # 这里将生成的噪声和真实类别进行计算损失，为了使得生成的图像更加接近真实图像。
        c_errG = c_criterion(c_output, c_label)
        errG = c_errG
        if train_generator:
            errG.backward()
            optimizerG.step()
        # errG.backward()
        D_G_z2 = c_output.data.mean()
        # optimizerG.step()
        right += correct
        # print('begin spout!')

    if epoch % 2 == 0:
        print('[%d/%d][%d/%d]   D(x): %.4f D(G(z)): %.4f / %.4f=%.4f,  Accuracy: %.4f / %.4f = %.4f'
              % (epoch, opt.niter, i, len(train_loader),
                 D_x, D_G_z1, D_G_z2, D_G_z1 / D_G_z2,
                 right, len(train_loader.dataset), 100. * right / len(train_loader.dataset)))

    # torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
    # torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))
    if epoch % 2 == 0:
        netD.eval()
        netG.eval()
        test_loss = 0
        right = 0
        all_Label = []
        all_target = []
        for data, target in test_loader:
            indx_target = target.clone()
            if opt.cuda:
                data, target = data.cuda(), target.cuda()
            with torch.no_grad():
                data, target = Variable(data), Variable(target)
            # batch_size = data.size(0)
            # noise.resize_(batch_size, nz, 1, 1)
            # noise.normal_(0, 1)
            # noise_ = np.random.normal(0, 1, (batch_size, nz, 1, 1))
            # noise.resize_(batch_size, nz, 1, 1).copy_(torch.from_numpy(noise_))

            # fake=netG(noise)
            # output = netD(data)
            # vutils.save_image(data,'%s/real_samples_i_%03d.png' % (opt.outf,epoch))
            # vutils.save_image(fake,'%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch))
            output = netD(data)

            test_loss += c_criterion(output, target).item()
            pred = output.max(1)[1]  # get the index of the max log-probability
            all_Label.extend(pred)
            all_target.extend(target)
            right += pred.cpu().eq(indx_target).sum()

        test_loss = test_loss / len(test_loader)  # average over number of mini-batch
        acc = float(100. * float(right)) / float(len(test_loader.dataset))
        print('\tTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
            test_loss, right, len(test_loader.dataset), acc))
        if acc > best_acc:
            best_acc = acc
            if opt.cuda:
                C = confusion_matrix([i.cpu() for i in all_target], [i.cpu() for i in all_Label])
                print(classification_report([i.cpu() for i in all_target], [i.cpu() for i in all_Label]))
            else:
                C = confusion_matrix(all_target, all_Label)
                print(classification_report(all_target, all_Label))
            C = C[:num_class, :num_class]
            # np.save('c.npy', C)
        if best_acc > 95:
            train_generator = False
        # np.save('c.npy', C)
        # print(C)
        k = kappa(C, np.shape(C)[0])
        AA_ACC = np.diag(C) / np.sum(C, 1)
        AA = np.mean(AA_ACC, 0)
        # print('OA= %.5f AA= %.5f k= %.5f' % (acc, AA, k))
        print('OA= %.5f AA= %.5f k= %.5f best_acc= %.5f' % (acc, AA, k, best_acc))