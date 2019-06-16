import os
import random
import pandas as pd
import numpy as np
import data_utils
from importlib import reload
reload(data_utils)

import matplotlib.pyplot as plt
from data_utils import prepare_daily_data
from data_utils import generate_tech_ind
from data_utils import generate_candlestick_ind

os.environ['PYTHONHASHSEED']=str(22)
random.seed(22)
np.random.seed(22)


import tensorflow as tf
from tensorflow import set_random_seed
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.recurrent import LSTM
from keras.layers import Conv1D, MaxPooling1D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import *
from keras.optimizers import RMSprop, Adam, SGD, Nadam
from keras import backend as K

from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


def create_Xt_Yt(X, y, percentage=0.9,test_index=None):
    p = int(len(X) * percentage)
    if(test_index != None):
        p=test_start_index
    X_train = X[0:p]
    Y_train = y[0:p]

    X_test = X[p:]
    Y_test = y[p:]

    return X_train, X_test, Y_train, Y_test

WINDOW = 30
EMB_SIZE = 7
STEP = 1


fcpo_data = pd.read_csv('data/fcpo_daily_2010_2018.csv')
fcpo_data= fcpo_data.set_index(pd.to_datetime(fcpo_data['Date']))
fcpo_data=fcpo_data[['Open','High','Low','Close','Volume']]
fcpo_data = prepare_daily_data(fcpo_data,long_spread_thr=1.5,short_spread_thr=1.5,lookup_period=4)

fcpo_tind = generate_tech_ind(fcpo_data.shift(1))
fcpo_model_data=fcpo_data[['Open','prev_1day_ret','prev_open_change_pct','lprofit_ind','sprofit_ind']].merge(fcpo_tind,left_index=True,right_index=True)
fcpo_model_data=fcpo_model_data.dropna()

## Normalize the data
fcpo_train_data=fcpo_model_data.drop(['lprofit_ind','sprofit_ind'],axis=1)
robust_scaler=RobustScaler()
fcpo_train_data=pd.DataFrame(robust_scaler.fit_transform(fcpo_train_data),index=fcpo_train_data.index,columns=fcpo_train_data.columns)

test_start_index=np.where(fcpo_train_data[WINDOW:].index=='2018-01-02')[0][0]

#np.column_stack(fcpo_model_data[30:34]['lprofit_ind'])
#fcpo_train_data[WINDOW:].head()

X,Y=[],[]
for i in range(WINDOW, len(fcpo_model_data)-4, STEP):
    try:
        atr = fcpo_train_data[i-WINDOW:i]['atr']
        adx = fcpo_train_data[i-WINDOW:i]['adx']
        cci = fcpo_train_data[i-WINDOW:i]['cci']
        macd = fcpo_train_data[i-WINDOW:i]['macd_close']
        rsi = fcpo_train_data[i-WINDOW:i]['rsi_close']
        prev_ret = fcpo_train_data[i-WINDOW:i]['prev_open_change_pct']
        stoch = fcpo_train_data[i-WINDOW:i]['stoch']

        x_i = np.column_stack((atr,adx,cci,macd,rsi,prev_ret,stoch))
        y_i =np.array(fcpo_model_data[i:i+4]['sprofit_ind'])

    except Exception as e:
        print(e)
        break
    X.append(x_i)
    Y.append(y_i)

X, Y = np.array(X), np.array(Y)
X_train, X_test, Y_train, Y_test = create_Xt_Yt(X, Y ,test_index=test_start_index)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], EMB_SIZE))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], EMB_SIZE))


def build_cnn_model(WINDOW,EMB_SIZE):
    cnn_model = Sequential()
    cnn_model.add(Conv1D(input_shape = (WINDOW, EMB_SIZE),
                            filters=16,
                            kernel_size=4,
                            padding='same'))
    cnn_model.add(ReLU())
    cnn_model.add(MaxPooling1D(2))
    cnn_model.add(Dropout(0.2))

    cnn_model.add(Conv1D(filters=32,
                            kernel_size=4,
                            padding='same'))
    cnn_model.add(ReLU())
    cnn_model.add(MaxPooling1D(2))
    cnn_model.add(Dropout(0.2))

    cnn_model.add(Flatten())
    #cnn_model.add(Dense(32))
    #cnn_model.add(LeakyReLU())

    #cnn_model.add(Dense(32))
    #cnn_model.add(LeakyReLU())

    cnn_model.add(Dense(4,activation='linear'))
    #cnn_model.add(Activation('softmax'))

    opt = Nadam(lr=0.002)

    #reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=25, min_lr=0.000001, verbose=1)
    #checkpointer = ModelCheckpoint(filepath="lolkekr.hdf5", verbose=1, save_best_only=True)
    cnn_model.compile(optimizer=opt,
                  loss='binary_crossentropy',metrics=['accuracy'])
    return cnn_model

def build_lstm_model(WINDOW,EMB_SIZE):
    lstm_model= Sequential()
    lstm_model.add(LSTM(64,input_shape=(WINDOW,EMB_SIZE),activation='relu',return_sequences=True))
    lstm_model.add(Dropout(0.2))
    #lstm_model.add(Flatten())
    lstm_model.add(LSTM(32,input_shape=(WINDOW,EMB_SIZE),activation='relu'))
    lstm_model.add(Dropout(0.2))
    lstm_model.add(Dense(16,activation='relu'))
    lstm_model.add(Dropout(0.2))
    lstm_model.add(Dense(1,activation='sigmoid'))
    lstm_model.compile(optimizer='adam',
                  loss='binary_crossentropy',metrics=['accuracy'])
    return lstm_model

model=build_cnn_model(WINDOW,EMB_SIZE)
history = model.fit(X_train, Y_train,
          epochs = 100,
          batch_size = 120,
          verbose=0,
          validation_data=(X_test, Y_test),
          #callbacks=[reduce_lr, checkpointer],
          shuffle=False)


predicted = model.predict(X_test)
#confusion_matrix(Y_test,pred_labels)
#recall_score(original,pred_labels)
#roc_auc_score(original,pred_labels)
#pd.DataFrame(np.column_stack((pred_labels,original)),columns=['predicted','original'],index=fcpo_data[test_start_index:].index)



plt.figure(figsize=(10,10))
plt.plot(history.history['loss'],label='loss')
plt.plot(history.history['val_loss'],label='val_loss')
plt.legend()
