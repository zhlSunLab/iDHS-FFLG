#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import matplotlib as mpl
mpl.use('Agg')

from data_load_processing import *
from model import DHS
from metrics_plot import *
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import gc
import _pickle as cPickle
from keras.preprocessing import sequence
from sklearn.model_selection import StratifiedKFold
from keras_self_attention import SeqSelfAttention,ScaledDotProductAttention
from keras.models import load_model

def funciton(DATAPATH, OutputDir):

    maxlen = 300

    X_train, y_train = createTrainData(DATAPATH)
    X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
    #X_train = sequence.pad_sequences(X_train, maxlen=maxlen, dtype='float32', padding='post')

    print('X_train shape:', X_train.shape)

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)


    trainning_result = []
    testing_result = []

    for i, (train, test) in enumerate(kfold.split(X_train, y_train)):

        print('\n\n%d' % i)

        model = DHS()

        early_stopping = EarlyStopping(monitor='val_loss', patience=10)
        model_check = ModelCheckpoint(filepath=OutputDir + "/model" + str(i + 1) + ".h5",
                                      monitor='val_loss', save_best_only=True)
        reduct_L_rate = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=8)
        model.fit(X_train[train], y_train[train], epochs=100, batch_size=128,
                  validation_data=(X_train[test], y_train[test]), shuffle=True, verbose=1,callbacks=[early_stopping,model_check,reduct_L_rate])
        filepath=OutputDir + "/model" + str(i + 1) + ".h5"
        model1 = load_model(filepath, custom_objects = {'SeqSelfAttention':SeqSelfAttention})

        trainning_result.append(calculateScore(X_train[train], y_train[train], model1,
                                               OutputDir + "/trainy_predy_" + str(i+1) + ".txt"))

        testing_result.append(calculateScore(X_train[test], y_train[test], model1,
                                             OutputDir + "/testy_predy_" + str(i+1) + ".txt"))

        del model
        gc.collect()

    temp_dict = (trainning_result, testing_result)
    analyze(temp_dict, OutputDir)

    del trainning_result, testing_result
    gc.collect()
