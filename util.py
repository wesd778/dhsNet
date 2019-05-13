from __future__ import division
from sklearn.model_selection import KFold
from random import Random
from tqdm import tqdm
random = Random()

import csv
import numpy as np


def data_normalization(species, chromosomes, path_seq_data, path_loci_dhs, path_normalization_data):
    A = 0
    B = 0
    L = []
    for chromosome in tqdm(chromosomes):
        Y = []
        N = []

        seq_name = path_seq_data + chromosome + '.fasta'
        DHSs_name = path_loci_dhs + 'dhss.gff3'

        File = open(path_normalization_data + species + '_normalization_data.fasta', "a")

        seq_file = csv.reader(open(seq_name, "r"))
        sequence = [row for row in seq_file]
        all_sequence_str = ""

        for i in sequence[1:]:
            all_sequence_str += i[0]

        DHSs = csv.reader(open(DHSs_name, "r"))
        a = [i[0].split('\t') for i in DHSs]
        seqL = np.zeros(len(all_sequence_str))

        for i in a:

            if species == 'Human':
                n1, n2 = 1, 2
            else:
                n1, n2 = 3, 4

            if i[0].lower() == chromosome:
                seqL[int(i[n1]):int(i[n2])] = 1
                j = int(i[n2]) - int(i[n1])

                if 200 <= j < 800:
                    A += 1
                    S = all_sequence_str[int(i[n1]):int(i[n2])]
                    L.append(j)
                    File.write('>Y' + str(A) + '_' + species + '_' + chromosome + '_' + i[n1] + ':' + i[n2] + '_' + str(j) + '\n')
                    File.write(S.upper() + '\n')
                    Y.append(S)

        for n in Y:
            x = 0

            while x < 1:
                j = len(n)
                pick = random.randint(0, len(all_sequence_str))
                S = all_sequence_str[pick - j//2:pick + j//2]

                if 1 not in seqL[pick - 2000:pick + 2000] and 'N' not in S and 200 <= len(S) < 800:
                    B += 1
                    x += 1
                    L.append(len(S))
                    File.write('>N' + str(B) + '_' + species + '_' + chromosome + '_' + str(pick - j//2) + ':' + str(pick + j//2) + '_' + str(j) + '\n')
                    File.write(S.upper() + '\n')
                    N.append(S)

        File.close()
    print "save " + path_normalization_data + species + '_normalization_data.fasta'
    print "NO. of Y is " + str(A), "NO. of N is " + str(B)
    print "seq_max = " + str(max(L)), "seq_min = " + str(min(L)), "seq_mean = " + str(np.mean(L))


def mul2gap(species, kfold, gap, chromosomes, path_seq_data, path_mul_train, path_gap_train, is_mul):
    if gap == 200:
        lens = ['200:400', '400:600', '600:800']
    elif gap == 100:
        lens = ['200:300', '300:400', '400:500', '500:600', '600:700', '700:800']
    else:
        lens = ['200:250', '300:350', '400:450', '500:550', '600:650', '700:750',
                '250:300', '350:400', '450:500', '550:600', '650:700', '750:800']

    if is_mul:
        for num in tqdm(range(kfold)):
        # for num in tqdm(range(1)):
            # File_mul_train = open(path_mul_train + species +'_mul_train.fasta', "r")
            File_mul_train = open(path_mul_train + species +'_mul_train' + str(num) + '.fasta', "r")
            N = [i for i in File_mul_train]
            for chromosome in chromosomes:
                File_gap_train = open(path_gap_train + species +'_gap'+ str(gap) +'_train' + str(num) + '.fasta', "a")
                # File_gap_train = open(path_gap_train + species +'_gap'+ str(gap) +'_train.fasta', "a")
                seq_name = path_seq_data + chromosome + '.fasta'
                seq_file = csv.reader(open(seq_name, "r"))
                sequence = [row for row in seq_file]
                all_sequence_str = ""

                for i in sequence[1:]:
                    all_sequence_str += i[0]

                for n in N:
                    if n[0] == '>':
                        i = n[:-1].split('_')
                        if i[2].lower() == chromosome:
                            j1, j2 = int(i[3].split(':')[0]), int(i[3].split(':')[1])

                            for len0 in lens:
                                l1, l2 = int(len0.split(':')[0]), int(len0.split(':')[1])
                                if l1 <= int(i[4]) < l2:
                                    if species == 'Human':
                                        j3 = l2 - (j2 - j1)
                                        for H in range(3):
                                            j4 = random.randint(0, j3)
                                            j5 = j3 - j4
                                            S = all_sequence_str[j1 - j4:j2 + j5]
                                            File_gap_train.write(i[0] + '_' + species + '_' + chromosome + '_'
                                                                 + str(j1 - j4) + ':' + str(j2 + j5) + '_' + str(l2) + '\n')
                                            File_gap_train.write(S.upper() + '\n')
                                    else:
                                        j3 = (j1 + j2)//2
                                        if l1 <= int(i[4]) < l2:
                                            S = all_sequence_str[j3 - l2//2:j3 + l2//2]
                                            File_gap_train.write(i[0] + '_' + species + '_' + chromosome + '_'
                                                                 + str(j3 - l2//2) + ':' + str(j3 + l2//2) + '_' + str(l2) + '\n')
                                            File_gap_train.write(S.upper() + '\n')

                File_gap_train.close()
            File_mul_train.close()

    else:
        for data in ['train', 'test']:
            for num in tqdm(range(kfold)):
                File_mul_train = open(path_mul_train + species + '_mul_' + data + str(num) + '.fasta', "r")
                N = [i for i in File_mul_train]

                for chromosome in chromosomes:
                    File_gap_train = open(path_gap_train + species + '_len' + str(gap) + '_' + data + str(num) + '.fasta', "a")

                    seq_name = path_seq_data + chromosome + '.fasta'
                    seq_file = csv.reader(open(seq_name, "r"))
                    sequence = [row for row in seq_file]
                    all_sequence_str = ""

                    for i in sequence[1:]:
                        all_sequence_str += i[0]

                    for n in N:
                        if n[0] == '>':
                            i = n[:-1].split('_')
                            if i[2] == chromosome:
                                j1, j2 = int(i[3].split(':')[0]), int(i[3].split(':')[1])
                                j3 = (j1 + j2)//2
                                S = all_sequence_str[j3 - gap//2:j3 + gap//2]
                                File_gap_train.write(i[0] + '_' + species + '_' + chromosome + '_'
                                                     + str(j3 - gap//2) + ':' + str(j3 + gap//2) + '_' + str(gap) + '\n')
                                File_gap_train.write(S.upper() + '\n')

                    File_gap_train.close()
                File_mul_train.close()


def kfold_split(species, kfold, path_normalization_data, path_mul_train, path_mul_test):
    # data_file = open(path_normalization_data, 'r')
    data_file = open(path_normalization_data + species + '_normalization_data.fasta', 'r')
    data = [i[:-1] for i in data_file]
    data_file.close()

    kf = KFold(n_splits=kfold, shuffle=True)

    for n, (train_index, test_index) in enumerate(kf.split(range(len(data)//2))):
        train_file = open(path_mul_train + species +'_mul_train' + str(n) + '.fasta', 'w')

        for i in train_index * 2:
            train_file.write(data[i] + '\n')
            train_file.write(data[i + 1] + '\n')

        train_file.close()

        test_file = open(path_mul_test + species +'_mul_test' + str(n) + '.fasta', 'w')

        for i in test_index * 2:
            test_file.write(data[i] + '\n')
            test_file.write(data[i + 1] + '\n')

        test_file.close()


def performance(labelArr, predictArr):
    TP, TN, FP, FN, SN, SP = 0., 0., 0., 0., 0., 0.

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
