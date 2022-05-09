#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from __future__ import print_function
from keras.models import load_model
import matplotlib as mpl
mpl.use('Agg')
import gc
from data_load_processing import *
from metrics_plot import *
import _pickle as cPickle
from keras.preprocessing import sequence
from sklearn import metrics
from sklearn.metrics import confusion_matrix, recall_score, matthews_corrcoef, \
    roc_curve, roc_auc_score, auc, precision_recall_curve
from self_attention import SeqSelfAttention

def funciton(DATAPATH, MODELPATH):
    maxlen = 300
    X_train, y_train = createTrainData(DATAPATH)
    X_train = sequence.pad_sequences(X_train, maxlen=maxlen)

    model = load_model(MODELPATH, custom_objects={'SeqSelfAttention':SeqSelfAttention})

    testing_result = []
    testing_result.append(calculateScore1(X_train, y_train, model))

    for j in ['sn', 'sp', 'acc', 'MCC', 'AUC', 'AUPR', 'precision', 'F1']:
        total = []
        for val in testing_result:
            total.append(val[j])
        print(j + ' : mean : ' + str(f'{np.mean(total): .3f}'))


def calculateScore1(X, y, model):
    """
    :param X         : data
    :param y         : labels
    :param model     : model
    :param OutputFile: output file about predict results
    :return          : metrics
    """
    score = model.evaluate(X, y)
    pred_y = model.predict(X)


    accuracy = score[1]

    tempLabel = [(0 if i < 0.5 else 1) for i in pred_y]
    confusion = confusion_matrix(y, tempLabel)

    TN, FP, FN, TP = confusion.ravel()
    sensitivity = recall_score(y, tempLabel)
    specificity = TN / float(TN + FP)
    MCC = matthews_corrcoef(y, tempLabel)
    F1Score = (2 * TP) / float(2 * TP + FP + FN)
    precision = TP / float(TP + FP)

    pred_y = pred_y.reshape((-1,))

    ROCArea = roc_auc_score(y, pred_y)
    fpr, tpr, thresholds = roc_curve(y, pred_y)

    precisionPR, recallPR, _ = precision_recall_curve(y, pred_y)
    aupr = auc(recallPR, precisionPR)

    return {'sn': sensitivity, 'sp': specificity, 'acc': accuracy, 'MCC': MCC, 'AUC': ROCArea, 'precision': precision,
            'F1': F1Score, 'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds, 'AUPR': aupr, 'precisionPR': precisionPR,
            'recallPR': recallPR, 'y_real': y, 'y_pred': pred_y}


if __name__ == "__main__":
    DATAPATH= r'D:\桌面\DATA/testdata/Neuraltube-test.fa'
    MODELPATH = r"D:\桌面\iDHS-FFLG\model\Neuraltube/model.h5"
    funciton(DATAPATH, MODELPATH)

