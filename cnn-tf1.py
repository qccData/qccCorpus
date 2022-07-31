###########################################################
# Adapted from: https://github.com/bagustris/deep_mlp_ser #
# (by Bagus Tris Atmaja, Masato Akagi)                    #
###########################################################


#%tensorflow_version 1.x 

import numpy as np
import pickle
import pandas as pd


import keras.backend as K
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Masking, TimeDistributed, \
                         Bidirectional, Flatten, Convolution1D, \
                         Embedding, Dropout, Flatten, BatchNormalization, \
                         RNN, concatenate, Activation

from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence

import random as rn
import tensorflow as tf


def calc_scores (x, y):
    # Computes the metrics CCC, PCC, and RMSE between the sequences x and y
    #  CCC:  Concordance correlation coeffient
    #  PCC:  Pearson's correlation coeffient
    #  RMSE: Root mean squared error
    # Input:  x,y: numpy arrays (one-dimensional)
    # Output: CCC,PCC,RMSE
    
    x_mean = np.nanmean(x)
    y_mean = np.nanmean(y)
    
    covariance = np.nanmean((x-x_mean)*(y-y_mean))
    
    x_var = 1.0 / (len(x)-1) * np.nansum((x-x_mean)**2) # Make it consistent with Matlab's nanvar (division by len(x)-1, not len(x)))
    y_var = 1.0 / (len(y)-1) * np.nansum((y-y_mean)**2)
    
    CCC = (2*covariance) / (x_var + y_var + (x_mean-y_mean)**2)
    
    x_std = np.sqrt(x_var)
    y_std = np.sqrt(y_var)
    
    PCC = covariance / (x_std * y_std)
    
    RMSE = np.sqrt(np.nanmean((x - y)**2))
    
    scores = np.array([ CCC, PCC, RMSE ])
    
    return scores

def CCC(y_true, y_pre):
    ccc = []
    for i in range(0, 3):
        ccc_, _, _ = calc_scores(y_pre[:, i], y_true[:, i])
        ccc.append(ccc_)
    
    return ccc


def load():
    x_test = np.load("sets/x_test.npy")
    x_train = np.load("sets/x_train.npy")
    x_val = np.load("sets/x_val.npy")
    y_test = np.load("sets/y_test.npy")
    y_train = np.load("sets/y_train.npy")
    y_val = np.load("sets/y_val.npy")
    return x_test, x_train, x_val, y_test, y_train, y_val


rn.seed(123)
np.random.seed(99)
tf.set_random_seed(1234)

# load feature and labels
x_test, x_train, x_val, y_test, y_train, y_val = load()

# for CNN input shape (batch, channel, steps)
x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])
x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1])
x_val = x_val.reshape(x_val.shape[0], 1, x_val.shape[1])



# Concordance correlation coefficient (CCC)-based loss function - using non-inductive statistics
def ccc(gold, pred):
    gold       = K.squeeze(gold, axis=-1)
    pred       = K.squeeze(pred, axis=-1)
    gold_mean  = K.mean(gold, axis=-1, keepdims=True)
    pred_mean  = K.mean(pred, axis=-1, keepdims=True)
    covariance = (gold-gold_mean)*(pred-pred_mean)
    gold_var   = K.mean(K.square(gold-gold_mean), axis=-1,  keepdims=True)
    pred_var   = K.mean(K.square(pred-pred_mean), axis=-1, keepdims=True)
    ccc        = K.constant(2.) * covariance / (gold_var + pred_var + K.square(gold_mean - pred_mean) + K.common.epsilon())
    return ccc


def ccc_loss(gold, pred):  
    # input (num_batches, seq_len, 1)
    ccc_loss   = K.constant(1.) - ccc(gold, pred)
    return ccc_loss

# API model, if use RNN, first two rnn layer must return_sequences=True
def api_model(alpha, beta, gamma):
    # speech network
    input_speech = Input(shape=(x_train.shape[1], x_train.shape[2]), name='speech_input')
    net_speech = BatchNormalization()(input_speech)
    net_speech = Convolution1D(256, 1, activation='relu')(net_speech)
    net_speech = Convolution1D(128, 1, activation='relu')(net_speech)
    net_speech = Convolution1D(64, 1, activation='relu')(net_speech)
    net_speech = Convolution1D(32, 1, activation='relu')(net_speech)
    net_speech = Convolution1D(16, 1, activation='relu')(net_speech)
    model_speech = Flatten()(net_speech)
    #model_speech = Dropout(0.1, seed=None)(net_speech)

    target_names = ('v', 'a', 'd')
    model_combined = [Dense(1, name=name)(model_speech) for name in target_names]
    #model_combined = Dense(3, activation='linear')(model_combined)
    
    model = Model(input_speech, model_combined) 
    #model.compile(loss=ccc_loss, optimizer='rmsprop', metrics=[ccc])
    model.compile(loss=ccc_loss, 
                  loss_weights={'v': alpha, 'a': beta, 'd': gamma},
                  optimizer='adam', metrics=[ccc])
    return model



#def main(alpha, beta, gamma):
model = api_model(0.1, 0.5, 0.4)
model.summary()


earlystop = EarlyStopping(monitor='val_loss', mode='min', patience=10,restore_best_weights=True)
hist = model.fit(x_train, y_train.T.tolist(), batch_size=34, 
                  validation_data=(x_val, y_val.T.tolist()), epochs=180, verbose=1, shuffle=True,
                  callbacks=[earlystop])
metrik = model.evaluate(x_test, y_test.T.tolist())

print(metrik)


predict = model.predict(x_test, batch_size=34)
list_=[]
for i in range(2484):
    list_.append([predict[0][i],predict[1][i],predict[2][i]])
list_=np.array(list_)
list_=list_.reshape(2484,3)
v = CCC(y_test, list_)
print(v)