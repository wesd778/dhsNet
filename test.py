from random import Random
from tqdm import tqdm
random = Random()
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef, roc_curve, auc
from sklearn.preprocessing import LabelBinarizer
from torch.autograd import Variable
from util import *

import torch
import torch.utils.data
import dhsNet
import numpy as np
import pandas as pd
import ConfigParser

specise = 'Arabidopsis'
config=ConfigParser.ConfigParser()
config.read('./config.txt')
net = dhsNet.MyNet()

kfold = int(config.get(specise, 'kfold'))
chromosomes = config.get(specise, 'chromosomes').split(';')
mul_train = config.get(specise, 'mul_train')
gap = int(config.get(specise, 'gap'))
output_file = config.get(specise, 'output_file')
batch = int(config.get(specise, 'batch'))
max_epochs = int(config.get(specise, 'max_epochs'))
test_name = config.get(specise, 'test_name').split(';')
model_name = config.get(specise, 'model_name').split(';')


def performance(labelArr, predictArr):
	TP = 0.; TN = 0.; FP = 0.; FN = 0.; SN = 0.; SP = 0.
	for i in range(len(labelArr)):
		if labelArr[i] == 1 and predictArr[i] == 1:
			TP += 1.
		if labelArr[i] == 1 and predictArr[i] == 0:
			FN += 1.
		if labelArr[i] == 0 and predictArr[i] == 1:
			FP += 1.
		if labelArr[i] == 0 and predictArr[i] == 0:
			TN += 1.
	if TP + FN == 0:
		SN = 0
	elif FP + TN == 0:
		SP = 0
	else:
		SN = TP/(TP + FN)
		SP = TN/(FP + TN)
	return SN, SP


def test(x_random, y_random):
	global batch_idx
	net.eval()
	y_pred = []
	y_true = []
	y_score = []
	for n in tqdm(range(len(x_random))):
		valid_seqs, valid_targets = x_random[n][np.newaxis,:,:,np.newaxis], y_random[n]
		valid_seqs = torch.FloatTensor(valid_seqs)
		valid_seqs = Variable(valid_seqs)
		valid_outputs = net(valid_seqs)
		a1,a2 = valid_outputs.data[0][0], valid_outputs.data[0][1]
		y_score.append(a2/(a1 + a2))
		if a1 > a2:
			y_pred.append(0)
		else:
			y_pred.append(1)
		y_true.append(valid_targets)
	valid_acc_bs = accuracy_score(y_true, y_pred)
	valid_mcc_bs = matthews_corrcoef(y_true, y_pred)
	valid_sn_bs, valid_sp_bs = performance(y_true, y_pred)
	fpr, tpr, threshold = roc_curve(y_true, y_score)
	valid_auc_bs = auc(fpr, tpr)
	print('valid_sn:{0}  valid_sp:{1}  valid_acc:{2}  valid_mcc:{3}  valid_auc:{4}'
		  .format(valid_sn_bs, valid_sp_bs, valid_acc_bs, valid_mcc_bs, valid_auc_bs))
	return fpr, tpr, valid_sn_bs, valid_sp_bs, valid_acc_bs, valid_mcc_bs, valid_auc_bs, y_pred


def data_load(data_name):
	S = []
	K = []
	File = open(data_name, "r")
	f = [i for i in File]
	File.close()

	for n in tqdm(range(0, len(f), 2)):
		if len([j for j in f[n+1][:-1]]) != 0:
			S0 = LabelBinarizer().fit_transform([j for j in f[n+1][:-1]]).T
		if S0.shape[0] == 4 and 200<=S0.shape[1]<=800:
			S.append(S0)
			if f[n][1] == 'Y':
				K.append(1)
			else:
				K.append(0)
	return S, K


results = [[specise, 'model_name', 'test_name', 'valid_sn', 'valid_sp', 'valid_acc', 'valid_mcc', 'valid_auc']]
for K in range(len(model_name)):
	net_name = './output/model/' + model_name[K]
	test_file = './output/' + specise + '/' + test_name[K]
	print (net_name)
	print (test_file)
	x_random, y_random = data_load(data_name=test_file)
	net = torch.load(net_name)

	fpr, tpr, valid_sn_bs, valid_sp_bs, valid_acc_bs, valid_mcc_bs, valid_auc_bs, valid_pred = test(x_random, y_random)

	File = open(test_file, "r")
	f = [i for i in File]
	File.close()

	test = []

	for i in range(len(valid_pred)):
		j = i*2
		if '>' in f[j]:
			test.append(f[j][:-1] + '_' + str(valid_pred[i]))
			test.append(f[j+1][:-1])

	with open('test.txt', 'w') as OUTPUT:
		for i in test:
			OUTPUT.write(i + '\n')

	result = [specise, model_name[K], test_name[K], valid_sn_bs, valid_sp_bs, valid_acc_bs, valid_mcc_bs, valid_auc_bs]
	results.append(result)

	np.save('fpr.npy', fpr)
	np.save('tpr.npy', tpr)

results = pd.DataFrame(results)

results.to_csv('./output/results_' + specise + '4.csv', header=False, index=False)


