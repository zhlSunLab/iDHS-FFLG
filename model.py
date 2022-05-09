#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from turtle import xcor
from tensorflow import keras
from keras.metrics import binary_accuracy
import warnings
from keras.optimizers import SGD, Adam
from keras.layers.recurrent import LSTM, GRU, SimpleRNN

warnings.filterwarnings("ignore")
from self_attention import SeqSelfAttention
from keras.layers import Dense, Dropout, Conv1D, Input, MaxPooling1D, Flatten, LeakyReLU, AveragePooling1D, concatenate, \
    Multiply, Bidirectional
from keras import regularizers
from keras.models import Input, Model
from keras.layers.core import Permute, Reshape, Dense, Lambda, K, RepeatVector, Flatten


def DHS():
    input_shape1 = (300, 23)
    inputs1 = Input(shape=input_shape1)
    x = Conv1D(filters=128, kernel_size=5, padding='same', kernel_regularizer=regularizers.l2(1e-3),
               bias_regularizer=regularizers.l2(1e-4), activation='relu', input_shape=input_shape1)(inputs1)

    x = MaxPooling1D(pool_size=2, strides=None, padding='valid')(x)
    x = Conv1D(filters=64, kernel_size=3, padding='same', kernel_regularizer=regularizers.l2(1e-3),
               bias_regularizer=regularizers.l2(1e-4), activation='relu', input_shape=input_shape1)(x)

    x = MaxPooling1D(pool_size=2, strides=None, padding='valid')(x)

    lstm_out = Bidirectional(LSTM(64, return_sequences=True), merge_mode='sum')(x)
    lstm_out = SeqSelfAttention(
        kernel_regularizer=regularizers.l2(1e-3),
        bias_regularizer=regularizers.l2(1e-4),
        attention_regularizer_weight=1e-4,
        attention_activation='sigmoid',
        name='Attention')(lstm_out)

    lstm_out = Dropout(0.2)(lstm_out)
    attention_mul = Flatten()(lstm_out)
    attention_mul = Dense(100, activation='relu', kernel_regularizer=regularizers.l2(1e-3),
                          bias_regularizer=regularizers.l2(1e-4))(attention_mul)
    outLayer = Dense(1, activation='sigmoid')(attention_mul)
    model = Model(inputs=[inputs1], outputs=outLayer)
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001),
                  metrics=['accuracy'])  # SGD(lr = 0.005, momentum=0.95) binary_accuracy Adam(lr=0.001)
    print(model.summary())
    return model




if __name__ == "__main__":
    DHS()