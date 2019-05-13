from __future__ import print_function
from __future__ import division
from random import Random
from tqdm import tqdm
random = Random()

from sklearn.metrics import accuracy_score, matthews_corrcoef, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from torch.autograd import Variable

from util import *

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import dhsNet
import numpy as np
import time
import ConfigParser




def initNetParams(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=1e-3)


def train_data_load(L, data_name):
    S = []
    K = []
    File = open(data_name, "r")
    f = [i[:-1] for i in File]
    File.close()

    for n in range(0, len(f), 2):
        str1 = f[n].split('_')
        S0 = LabelBinarizer().fit_transform([j for j in f[n + 1]]).T
        if S0.shape[0] == 4 and str(L) == str1[4] == str(S0.shape[1]):
            S.append(S0)
            if 'Y' in str1[0]:
                K.append(1)
            else:
                K.append(0)
    return S, K


def train(epoch, x_number, y_number):
    global batch_idx
    net.train()

    mylr = 0.02 * (0.97 ** epoch)
    optimizer = optim.SGD(net.parameters(), lr=mylr, momentum=0.98, weight_decay=5e-4)
    train_data = torch.utils.data.TensorDataset(torch.from_numpy(x_number).float(), torch.from_numpy(y_number))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch, shuffle=True, num_workers=2)

    for batch_idx, (train_seqs, train_targets) in enumerate(train_loader):
        if use_cuda:
            train_seqs, train_targets = train_seqs.cuda(), train_targets.cuda()
        train_seqs, train_targets = Variable(train_seqs), Variable(train_targets)

        optimizer.zero_grad()
        train_outputs = net(train_seqs)
        loss = criterion(train_outputs, train_targets)
        loss.backward()
        optimizer.step()


def valid(x_number, y_number):
    global batch_idx
    net.eval()

    y_pred = []
    y_true = []
    y_loss = 0
    valid = torch.utils.data.TensorDataset(torch.from_numpy(x_number).float(), torch.from_numpy(y_number))
    validloader = torch.utils.data.DataLoader(valid, batch_size=batch, shuffle=True, num_workers=2)

    for batch_idx, (valid_seqs, valid_targets) in enumerate(validloader):

        if use_cuda:
            valid_seqs, valid_targets = valid_seqs.cuda(), valid_targets.cuda()

        valid_seqs, valid_targets = Variable(valid_seqs, volatile=True), Variable(valid_targets)
        valid_outputs = net(valid_seqs)
        loss = criterion(valid_outputs, valid_targets)
        _, predicted = torch.max(valid_outputs.data, 1)
        y_pred.extend(predicted.tolist())
        y_true.extend(valid_targets.data.tolist())
        y_loss += loss.data.cpu().numpy()

    return y_true, y_pred, y_loss


config=ConfigParser.ConfigParser()
config.read('./config.txt')
species = 'Arabidopsis'
kfold = int(config.get(species, 'kfold'))
chromosomes = config.get(species, 'chromosomes').split(';')
mul_train = config.get(species, 'mul_train')
output_file = config.get(species, 'output_file')
batch = int(config.get(species, 'batch'))
max_epochs = int(config.get(species, 'max_epochs'))

gaps = [100]
for gap in gaps:
    for K in range(kfold):
        path_train_file = output_file + species + '_gap' + str(gap) + '_train' + str(K) + '.fasta'
        if mul_train:
            if gap == 200:
                lens = ['400', '600', '800']
            elif gap == 100:
                lens = ['300', '400', '500', '600', '700', '800']
            else:
                lens = ['250', '350', '450', '550', '650', '750', '300', '400', '500', '600', '700', '800']
        else:
            lens = [str(gap)]


        x = {}
        y = {}

        print('#####################################')
        print('load data:')
        print(species, chromosomes, mul_train, gap, batch, max_epochs, path_train_file)

        for L in lens:
            x[L], y[L] = train_data_load(L=L, data_name=path_train_file)
            x[L], y[L] = np.array(x[L])[:, :, :, np.newaxis], np.array(y[L])
            print ('lens_' + str(L), x[L].shape, y[L].shape)

        net = dhsNet.MyNet()
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            net.cuda()
            net = torch.nn.DataParallel(net)
            cudnn.benchmark = True
        criterion = nn.CrossEntropyLoss()
        initNetParams(net)

        print('#####################################')
        print('training')
        epoch = 1
        T0 = time.time()
        while epoch <= max_epochs:
            print('\nEpoch: %d' % epoch)
            valid_loss = 0
            y_true, y_pred = [], []

            for L in tqdm(lens):
                train(epoch, x[L], y[L])
                true, pred, y_loss = valid(x[L], y[L])
                y_true.extend(true)
                y_pred.extend(pred)
                valid_loss += y_loss/128

            valid_acc = accuracy_score(y_true, y_pred)
            valid_mcc = matthews_corrcoef(y_true, y_pred)
            valid_confusion = confusion_matrix(y_true, y_pred)
            valid_sn, valid_sp = performance(y_true, y_pred)

            Y_num = sum(y_true)
            N_num = len(y_true) - Y_num

            print ('Y_num:' + str(Y_num) + '    N_num:' + str(N_num))
            print('valid_loss:{0}  valid_sn: {1}  valid_sp:{2}  valid_acc: {3}  valid_mcc: {4}'.format(valid_loss/len(lens), valid_sn, valid_sp, valid_acc, valid_mcc))
            print ('confusion_matrix:')
            print (valid_confusion)

            epoch += 1
        T1 = time.time()
        print ('finish')
        print ('cost  ' + str(T1-T0) + '  seconds')
        print ('#####################################')

        torch.save(net, './output/model/' + species + str(K) + '_' + str(gap) + '_' + str(int(T1-T0)) + 's.pkl')









