#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import random
import numpy as np
import _pickle as cPickle
from keras.preprocessing import sequence
import numpy as np
import collections
from sklearn.model_selection import train_test_split

def dataProcessing(seq):
    #################### 2222222222222222222222222222222  ############################
    chem_bases = {'A': [1, 1, 1], 'C': [0, 1, 0], 'G': [1, 0, 0, ], 'T': [0, 0, 1]}
    Z = np.zeros((len(seq), 3))
    for l, s in enumerate(seq):
        for i, char in enumerate(s):
            if char in chem_bases:
                Z[l] = (chem_bases[char])
    return Z

def get_1_trids():
    nucle_com = []
    chars = ['A', 'C', 'G', 'T']
    base = len(chars)
    end = len(chars) ** 1
    for i in range(0, end):
        n = i
        ch0 = chars[n % base]
        nucle_com.append(ch0)
    word_index = dict((w, i) for i, w in enumerate(nucle_com))
    #print(word_index)
    return word_index


def get_2_trids():
    nucle_com = []
    chars = ['A', 'C', 'G', 'T']
    base = len(chars)
    end = len(chars) ** 2
    for i in range(0, end):
        n = i
        ch0 = chars[n % base]
        n = n // base
        ch1 = chars[n % base]
        nucle_com.append(ch1 + ch0)
    word_index = dict((w, i) for i, w in enumerate(nucle_com))
    #print(word_index)
    return word_index

def get_3_trids():
    nucle_com = []
    chars = ['A', 'C', 'G', 'T']
    base = len(chars)
    end = len(chars) ** 3
    for i in range(0, end):
        n = i
        ch0 = chars[n % base]
        n = n // base
        ch1 = chars[n % base]
        n = n // base
        ch2 = chars[n % base]
        nucle_com.append(ch0 + ch1 + ch2)
    word_index = dict((w, i) for i, w in enumerate(nucle_com))
    return word_index


def get_4_trids():
    nucle_com = []
    chars = ['A', 'C', 'G', 'T']
    base = len(chars)
    end = len(chars) ** 4
    for i in range(0, end):
        n = i
        ch0 = chars[n % base]
        n = n // base
        ch1 = chars[n % base]
        n = n // base
        ch2 = chars[n % base]
        n = n // base
        ch3 = chars[n % base]
        nucle_com.append(ch0 + ch1 + ch2 + ch3)
    word_index = dict((w, i) for i, w in enumerate(nucle_com))
    return word_index


def frequency(seq, kmer, coden_dict):
    Value = []
    k = kmer
    coden_dict = coden_dict
    for i in range(len(seq) - int(k) + 1):
        kmer = seq[i:i + k]
        kmer_value = coden_dict[kmer]
        Value.append(kmer_value)
    freq_dict = dict(collections.Counter(Value))
    return freq_dict


def coden(seq, kmer, tris):
    coden_dict = tris
    freq_dict = frequency(seq, kmer, coden_dict)
    vectors = np.zeros((len(seq), len(coden_dict.keys())))
    for i in range(len(seq) - int(kmer) + 1):
        value = freq_dict[coden_dict[seq[i:i + kmer]]]
        vectors[i][coden_dict[seq[i:i + kmer]]] = 1
    return vectors


def get_RNA_seq_concolutional_array(seq, motif_len=4):
    print(seq)
    alpha = 'ACGT'
    row = (len(seq) + 2 * motif_len - 2)
    new_array = np.zeros((row, 4))
    for i in range(motif_len - 1):
        new_array[i] = np.array([0.25] * 4)

    for i in range(row - 3, row):
        new_array[i] = np.array([0.25] * 4)

    # pdb.set_trace()
    for i, val in enumerate(seq):
        i = i + motif_len - 1
        if val not in 'ACGT':
            new_array[i] = np.array([0.25] * 4)
            continue
        try:
            index = alpha.index(val)
            new_array[i][index] = 1
        except:
            pdb.set_trace()
    print(new_array)
    return new_array


def dealwithdata(sequence):
    tris1 = get_1_trids()
    tris2 = get_2_trids()
    tris3 = get_3_trids()
    dataX = []
    Z = dataProcessing(sequence)
    kmer1 = coden(sequence, 1, tris1)
    kmer2 = coden(sequence, 2, tris2)
    kmer3 = coden(sequence, 3, tris3)
    Kmer = np.hstack((kmer1,Z,kmer2))
    # dataX.append(Kmer.tolist())
    # dataX = np.array(dataX)
    # print(dataX.shape)
    return Kmer



def createTrainData(str1):
    sequence_num = []
    label_num = []
    f = open(str1).readlines()
    for i in range(0,len(f)-1,2):
        label = f[i].strip('\n').replace('>','')
        label_num.append(int(label))
        sequence = f[i+1].strip('\n')
        sequence_num.append(dealwithdata(sequence))

    X_train = sequence_num
    labels = label_num
    seed = 113
    np.random.seed(seed)
    np.random.shuffle(X_train)
    np.random.seed(seed)
    np.random.shuffle(labels)
    X_train = np.array(X_train)
    y_train = np.array(labels)

    return X_train, y_train



if __name__ == "__main__":
    X_test,y_test = createTrainData('./train.fa')
    #X_test = sequence.pad_sequences(X_test, maxlen=300, dtype='float32', padding='post')
    #X_test = np.around(X_test, 4)
    print('X_test shape:', X_test)
    print(y_test.shape)

