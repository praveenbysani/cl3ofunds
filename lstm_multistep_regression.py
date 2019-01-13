import os
os.chdir('/Users/praveen/Documents/Pythonspace/futures_trading/')
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.recurrent import LSTM, GRU
from keras.layers import Conv1D, MaxPooling1D, AtrousConvolution1D, RepeatVector
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from keras.layers.wrappers import Bidirectional
from keras import regularizers
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import *
from keras.optimizers import RMSprop, Adam, SGD, Nadam
from keras.initializers import *
from keras.constraints import *
from sklearn.preprocessing import RobustScaler
from keras import backend as K
import seaborn as sns
sns.despine()


def shuffle_in_unison(a, b):
    # courtsey http://stackoverflow.com/users/190280/josh-bleecher-snyder
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b


def create_Xt_Yt(X, y, percentage=0.9):
    p = int(len(X) * percentage)
    X_train = X[0:p]
    Y_train = y[0:p]

    X_train, Y_train = shuffle_in_unison(X_train, Y_train)

    X_test = X[p:]
    Y_test = y[p:]

    return X_train, X_test, Y_train, Y_test

epsilon = 1.0e-9
def qlike_loss(y_true, y_pred):
    y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
    loss = K.log(y_pred) + y_true / y_pred
    return K.mean(loss, axis=-1)


def mse_log(y_true, y_pred):
    y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
    loss = K.square(K.log(y_true) - K.log(y_pred))
    return K.mean(loss, axis=-1)


def mse_sd(y_true, y_pred):
    y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
    loss = K.square(y_true - K.sqrt(y_pred))
    return K.mean(loss, axis=-1)


def hmse(y_true, y_pred):
    y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
    loss = K.square(y_true / y_pred - 1.)
    return K.mean(loss, axis=-1)


def stock_loss(y_true, y_pred):
    alpha = 100.
    loss = K.switch(K.less(y_true * y_pred, 0), \
        alpha*y_pred**2 - K.sign(y_true)*y_pred + K.abs(y_true), \
        K.abs(y_true - y_pred)
        )
    return K.mean(loss, axis=-1)

def data2change(data):
    change = pd.DataFrame(data).pct_change()
    change = change.replace([np.inf, -np.inf], np.nan)
    change = change.fillna(0.).values.tolist()
    change = [c[0] for c in change]
    return change

def rolling_zscore(x, window):
    r = x.rolling(window=window)
    m = r.mean().shift(1)
    s = r.std(ddof=0).shift(1)
    z = (x-m)/s
    return z.bfill()


WINDOW = 30
EMB_SIZE = 5
STEP = 1
FORECAST = 5

data_original = pd.read_csv('./data/fcpo_daily_2018.csv')

openp = data_original.loc[:, 'Open']
highp = data_original.loc[:, 'High']
lowp = data_original.loc[:, 'Low']
closep = data_original.loc[:, 'Close']
volumep = data_original.loc[:, 'Volume']

openp_chng=data2change(openp)
highp_chng=data2change(highp)
lowp_chng=data2change(lowp)
closep_chng=data2change(closep)
volumep_chng=data2change(volumep)

openp_norm = rolling_zscore(openp,WINDOW).tolist()
highp_norm = rolling_zscore(highp,WINDOW).tolist()
lowp_norm = rolling_zscore(lowp,WINDOW).tolist()
closep_norm = rolling_zscore(closep,WINDOW).tolist()
volumep_norm = rolling_zscore(volumep,WINDOW).tolist()

volatility = []
for i in range(WINDOW, len(openp_chng)):
    window = closep[i-WINDOW:i]
    volatility.append(np.std(window))
#rscaler=RobustScaler()
#volatility=rscaler.fit_transform(pd.DataFrame(volatility)).reshape(1,-1)[0]

#daily_returns = []
#for i in range(0, len(data_original)):
#    return_pct = (closep[i] - openp[i])/(openp[i])
#    daily_returns.append(np.clip(return_pct,-2,2))

#plt.plot(pd.Series(highp_chng).rolling(WINDOW).sum())
openp_norm, highp_norm, lowp_norm, closep_norm, volumep_norm = openp_norm[WINDOW:], highp_norm[WINDOW:], lowp_norm[WINDOW:], closep_norm[WINDOW:], volumep_norm[WINDOW:]
X, Y = [], []
for i in range(0, len(openp_norm)-WINDOW-FORECAST, STEP):
    try:
        o = openp_norm[i:i+WINDOW]
        h = highp_norm[i:i+WINDOW]
        l = lowp_norm[i:i+WINDOW]
        c = closep_norm[i:i+WINDOW]
        v = volumep_norm[i:i+WINDOW]

        #volat = volatility[i:i+WINDOW]
        #r = daily_returns[i:i+WINDOW]
        x_i = np.column_stack(( o, h, l, c, v))
        y_i =np.array(closep_norm[i+WINDOW:i+WINDOW+FORECAST])
        #y_i =closep_norm[i+WINDOW+FORECAST]
        #x_i = c
    except Exception as e:
        print(e)
        break
    X.append(x_i)
    Y.append(y_i)


X, Y = np.array(X), np.array(Y)
X_train, X_test, Y_train, Y_test = create_Xt_Yt(X, Y)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], EMB_SIZE))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], EMB_SIZE))

def build_cnn_model(WINDOW,EMB_SIZE):
    cnn_model = Sequential()
    cnn_model.add(Conv1D(input_shape = (WINDOW, EMB_SIZE),
                            filters=16,
                            kernel_size=4,
                            padding='same'))
    cnn_model.add(MaxPooling1D(2))
    cnn_model.add(LeakyReLU())
    cnn_model.add(Conv1D(filters=32,
                            kernel_size=4,
                            padding='same'))
    cnn_model.add(MaxPooling1D(2))
    cnn_model.add(LeakyReLU())
    cnn_model.add(Flatten())

    cnn_model.add(Dense(32))
    cnn_model.add(LeakyReLU())

    cnn_model.add(Dense(2))
    cnn_model.add(Activation('linear'))

    opt = Nadam(lr=0.002)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=25, min_lr=0.000001, verbose=1)
    checkpointer = ModelCheckpoint(filepath="lolkekr.hdf5", verbose=1, save_best_only=True)

    cnn_model.compile(optimizer=opt,
                  loss=mse_log)
    return cnn_model

def build_lstm_model(WINDOW,EMB_SIZE,FORECAST):
    lstm_model= Sequential()
    lstm_model.add(LSTM(64,input_shape=(WINDOW,EMB_SIZE),activation='relu',return_sequences=True))
    lstm_model.add(Dropout(0.2))
    #lstm_model.add(Flatten())
    lstm_model.add(LSTM(32,input_shape=(WINDOW,EMB_SIZE),activation='relu'))
    lstm_model.add(Dropout(0.2))
    lstm_model.add(Dense(16,activation='relu'))
    lstm_model.add(Dropout(0.2))
    lstm_model.add(Dense(FORECAST,activation='linear'))
    lstm_model.compile(optimizer='adam',
                  loss='mse')
    return lstm_model

model=build_lstm_model(WINDOW,EMB_SIZE,FORECAST)

try:
    history = model.fit(X_train, Y_train,
              epochs = 30,
              batch_size = 256,
              verbose=1,
              validation_data=(X_test, Y_test),
              #callbacks=[reduce_lr, checkpointer],
              shuffle=True)
except Exception as e:
    print e
finally:
#    model.load_weights("lolkekr.hdf5")
    pred = model.predict(X_test)
    predicted = pred
    original = Y_test
    plt.figure(figsize=(10,10))
    plt.title('Actual and predicted')
    plt.plot(original[10], color='black', label = 'Original data')
    plt.plot(predicted[10], color='blue', label = 'Predicted data')
    plt.legend(loc='best')
    plt.show()
    print( np.mean(np.square(predicted - original)))
    print (np.mean(np.abs(predicted - original)))
    print( np.mean(np.abs((original - predicted) / original)))
