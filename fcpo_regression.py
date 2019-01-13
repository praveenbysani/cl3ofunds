import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

import random
random.seed(333)
np.random.seed(333)

import ta
import talib
from tech_indicators import stoch,stoch_signal

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RandomizedSearchCV

import xgboost


def compute_long_spread_v2(x):
    return round((x['next_nhigh']-x['Open'])/x['Open'],4)*100

def compute_short_spread_v2(x):
    return round((x['Open']-x['next_nlow'])/x['Open'],4)*100


def prepare_daily_data(fcpo_data,long_spread_thr=2,short_spread_thr=2,lookup_period=5):
    # extract the highest value over the next n days and the corresponding day index
    dayn_max_value_list=[]
    for i in range(lookup_period):
        dayn_max_value_list.append(fcpo_data['High'].shift(-i))
    dayn_max_df=pd.concat(dayn_max_value_list,axis=1)
    dayn_max_df.columns = list(range(lookup_period))
    maxhigh_values=dayn_max_df.max(axis=1)
    maxhigh_indexes=dayn_max_df.idxmax(axis=1)

    # extract the lowest value over the next n days and the corresponding day index
    dayn_min_value_list=[]
    for i in range(lookup_period):
        dayn_min_value_list.append(fcpo_data['Low'].shift(-i))
    dayn_min_df=pd.concat(dayn_min_value_list,axis=1)
    dayn_min_df.columns = list(range(lookup_period))
    minlow_values=dayn_min_df.min(axis=1)
    minlow_indexes=dayn_min_df.idxmin(axis=1)

    close_values=fcpo_data['Close'].shift(np.negative(lookup_period-1))

    fcpo_data=fcpo_data.assign(
    next_nhigh=maxhigh_values,
    next_nhigh_idx=maxhigh_indexes,
    next_nlow=minlow_values,
    next_nlow_idx=minlow_indexes,
    next_nclose=close_values,
    prev_1open=fcpo_data['Open'].shift(1),
    prev_1high=fcpo_data['High'].shift(1),
    prev_1low=fcpo_data['Low'].shift(1),
    prev_1close=fcpo_data['Close'].shift(1),
    )
    fcpo_data=fcpo_data.assign(
                                prev_open_change_pct=(fcpo_data['Open']/fcpo_data['prev_1close'])*100-100,
                                prev_1day_ret=(fcpo_data['prev_1close']/fcpo_data['prev_1open'])*100-100,
                                long_spread = fcpo_data.apply(lambda x: compute_long_spread_v2(x),axis=1),
                                short_spread = fcpo_data.apply(lambda x: compute_short_spread_v2(x),axis=1)
                            )
    fcpo_data=fcpo_data.assign(long_short_spread_diff= fcpo_data['long_spread']-fcpo_data['short_spread'])
    fcpo_data['prev_open_change_pct']=fcpo_data['prev_open_change_pct'].bfill()
    return fcpo_data

def zscore_func_improved(x,window_size=20):
    rolling_mean=x.rolling(window=window_size).mean().bfill()
    rolling_std = x.rolling(window=window_size).std().bfill()
    return (x-rolling_mean)


def generate_tech_ind(fcpo_df):
    fcpo_df_tind=pd.DataFrame({
        #'ema_slow_open':ta.trend.ema_slow(fcpo_df['Open'],n_slow=25).bfill(),
        #'ema_fast_open':ta.trend.ema_fast(fcpo_df['Open'],n_fast=10).bfill(),
        #'sma_slow_open': talib.SMA(fcpo_df['Open'],timeperiod=25).bfill(),
        #'sma_fast_open': talib.SMA(fcpo_df['Open'],timeperiod=10).bfill(),
        #'macd_open': ta.trend.macd(fcpo_df['Open']).bfill(),
        #'macd_signal_open': ta.trend.macd_signal(fcpo_df['Open']).bfill(),
        #'rsi_open':ta.momentum.rsi(fcpo_df['Open']).bfill(),
        #'ema_slow_high':ta.trend.ema_slow(fcpo_df['High'],n_slow=25).bfill(),
        #'ema_fast_high':ta.trend.ema_fast(fcpo_df['High'],n_fast=10).bfill(),
        #'sma_slow_high': talib.SMA(fcpo_df['High'],timeperiod=25).bfill(),
        #'sma_fast_high': talib.SMA(fcpo_df['High'],timeperiod=10).bfill(),
        #'macd_high': ta.trend.macd(fcpo_df['High']).bfill(),
        #'macd_signal_high': ta.trend.macd_signal(fcpo_df['High']).bfill(),
        #'rsi_high':ta.momentum.rsi(fcpo_df['High']).bfill(),
        #'ema_slow_low':ta.trend.ema_slow(fcpo_df['Low'],n_slow=25).bfill(),
        #'ema_fast_low':ta.trend.ema_fast(fcpo_df['Low'],n_fast=10).bfill(),
        #'sma_slow_low': talib.SMA(fcpo_df['Low'],timeperiod=25).bfill(),
        #'sma_fast_low': talib.SMA(fcpo_df['Low'],timeperiod=10).bfill(),
        #'macd_low': ta.trend.macd(fcpo_df['Low']).bfill(),
        #'macd_signal_low': ta.trend.macd_signal(fcpo_df['Low']).bfill(),
        #'rsi_low':ta.momentum.rsi(fcpo_df['Low']).bfill(),
        'ema_slow_close':ta.trend.ema_slow(fcpo_df['Close'],n_slow=25).bfill(),
        'ema_fast_close':ta.trend.ema_fast(fcpo_df['Close'],n_fast=10).bfill(),
        #'sma_slow_close': talib.SMA(fcpo_df['Close'],timeperiod=25).bfill(),
        #'sma_fast_close': talib.SMA(fcpo_df['Close'],timeperiod=10).bfill(),
        'macd_close': ta.trend.macd(fcpo_df['Close']).bfill(),
        'macd_signal_close': ta.trend.macd_signal(fcpo_df['Close']).bfill(),
        'rsi_close':ta.momentum.rsi(fcpo_df['Close']).bfill(),
        'stoch' : stoch(fcpo_df['High'],fcpo_df['Low'],fcpo_df['Close']).bfill(),
        'stoch_signal' : stoch_signal(fcpo_df['High'],fcpo_df['Low'],fcpo_df['Close']).bfill(),
        'adx': ta.trend.adx(fcpo_df['High'],fcpo_df['Low'],fcpo_df['Close']).bfill(),
        'atr': ta.volatility.average_true_range(fcpo_df['High'],fcpo_df['Low'],fcpo_df['Close']).bfill(),
        #'find' : ta.volume.force_index(fcpo_df['Close'],fcpo_df['Volume']).bfill(),
        'eom': ta.volume.ease_of_movement(fcpo_df['High'],fcpo_df['Low'],fcpo_df['Close'],fcpo_df['Volume'],n=10,fillna=True).bfill()
       })
    return fcpo_df_tind

def generate_candlestick_ind(fcpo_df):
    fcpo_cdl_ind=pd.DataFrame({'cdl_rickshawman':talib.CDLRICKSHAWMAN(fcpo_df['Open'],fcpo_df['High'],fcpo_df['Low'],fcpo_df['Close']),
                  'cdl_longlegdoji':talib.CDLLONGLEGGEDDOJI(fcpo_df['Open'],fcpo_df['High'],fcpo_df['Low'],fcpo_df['Close']),
                  'cdl_harami': talib.CDLHARAMI(fcpo_df['Open'],fcpo_df['High'],fcpo_df['Low'],fcpo_df['Close']),
                  'cdl_spintop': talib.CDLSPINNINGTOP(fcpo_df['Open'],fcpo_df['High'],fcpo_df['Low'],fcpo_df['Close']),
                  'cdl_marubozu': talib.CDLMARUBOZU(fcpo_df['Open'],fcpo_df['High'],fcpo_df['Low'],fcpo_df['Close']),
                  'cdl_longline': talib.CDLLONGLINE(fcpo_df['Open'],fcpo_df['High'],fcpo_df['Low'],fcpo_df['Close']),
                  'cdl_hikkake': talib.CDLHIKKAKE(fcpo_df['Open'],fcpo_df['High'],fcpo_df['Low'],fcpo_df['Close']),
                  'cdl_highwave': talib.CDLHIGHWAVE(fcpo_df['Open'],fcpo_df['High'],fcpo_df['Low'],fcpo_df['Close']),
                  'cdl_engulfing': talib.CDLENGULFING(fcpo_df['Open'],fcpo_df['High'],fcpo_df['Low'],fcpo_df['Close']),
                  'cdl_doji': talib.CDLDOJI(fcpo_df['Open'],fcpo_df['High'],fcpo_df['Low'],fcpo_df['Close']),
                  'cdl_closingmarubozu': talib.CDLCLOSINGMARUBOZU(fcpo_df['Open'],fcpo_df['High'],fcpo_df['Low'],fcpo_df['Close']),
                  'cdl_belthold': talib.CDLBELTHOLD(fcpo_df['Open'],fcpo_df['High'],fcpo_df['Low'],fcpo_df['Close'])
                   })
    return fcpo_cdl_ind


def build_regression_model_data(fcpo_data_daily,fcpo_feats,target_label='long_short_spread_diff',
                        train_start='2014-01-01',train_end='2017-12-31',test_start='2018-01-01',test_end='2018-10-01'):
    fcpo_train=fcpo_feats[train_start:train_end]
    fcpo_test = fcpo_feats[test_start:test_end]

    fcpo_train=fcpo_train.merge(fcpo_data_daily[['Open','prev_1day_ret','prev_open_change_pct',target_label]],left_index=True,right_index=True)
    fcpo_traindata=fcpo_train.iloc[:,0:-1]
    fcpo_trainlabels=fcpo_train.iloc[:,-1:]

    fcpo_test=fcpo_test.merge(fcpo_data_daily[['Open','prev_1day_ret','prev_open_change_pct',target_label]],left_index=True,right_index=True)
    fcpo_testdata=fcpo_test.iloc[:,0:-1]
    fcpo_testlabels=fcpo_test.iloc[:,-1]
    return fcpo_traindata,fcpo_trainlabels,fcpo_testdata,fcpo_testlabels

if __name__=='__main__':
    fcpo_data=pd.read_csv('data/fcpo_daily_2018.csv')
    fcpo_data=fcpo_data[['Date','Open','High','Low','Close','Volume']]
    fcpo_data=fcpo_data.set_index(pd.to_datetime(fcpo_data['Date']))
    fcpo_data=fcpo_data.drop(columns=['Date'])

    fcpo_data_daily=prepare_daily_data(fcpo_data,lookup_period=1)
    fcpo_daily_tind=generate_tech_ind(fcpo_data_daily[['Open','High','Low','Close','Volume']].shift(1))
    fcpo_daily_cdlind=generate_candlestick_ind(fcpo_data_daily[['Open','High','Low','Close']].shift(1))
    fcpo_feats=fcpo_daily_tind.merge(fcpo_daily_cdlind,left_index=True,right_index=True)
    #TODO : drop highly correlated features

    fcpo_traindata,fcpo_trainlabels,fcpo_testdata,fcpo_testlabels=build_regression_model_data(
                            fcpo_data_daily,fcpo_feats,'long_short_spread_diff')

    from sklearn.preprocessing import RobustScaler, MinMaxScaler
    robust_scaler_labels=RobustScaler()
    robust_scaler_data=RobustScaler()

    fcpo_scaled_trainlabels=robust_scaler_labels.fit_transform(fcpo_trainlabels)
    fcpo_scaled_traindata=robust_scaler_data.fit_transform(fcpo_traindata)

    fcpo_scaled_testlabels=pd.Series(robust_scaler.transform(np.array(fcpo_testlabels).reshape(1,-1))[0],
                                            index=fcpo_testlabels.index)
    fcpo_scaled_testdata=robust_scaler_data.transform(fcpo_testdata)
   '''
    xgb_param_dist={
               "n_estimators": [200,300,500],
               "max_depth": [5,10,20],
               "learning_rate": [0.01,0.05,0.1],
               "gamma": [0,0.01,0.05],
               }

    xgb_rsearch=RandomizedSearchCV(xgboost.XGBRegressor(random_state=333),
                           xgb_param_dist,random_state=333,cv=fcpo_tsplit,scoring='neg_mean_absolute_error',n_iter=100,n_jobs=-1)
    xgb_rsearch.fit(fcpo_traindata,np.ravel(fcpo_trainlabels))
    mean_absolute_error(fcpo_testlabels,xgb_regressor.predict(fcpo_testdata))

    fcpo_tsplit=TimeSeriesSplit(n_splits=3)
    fcpo_cval_scores=cross_val_score(xgb_regressor,fcpo_traindata,np.ravel(fcpo_trainlabels),
                                       scoring='neg_mean_squared_error',cv=fcpo_tsplit)

    '''

    xgb_regressor=xgboost.XGBRegressor(n_estimators=200,max_depth=5,learning_rate=0.01,gamma=0.05)


    xgb_regressor=xgb_regressor.fit(fcpo_traindata,fcpo_trainlabels)
    fcpo_pred_values=xgb_regressor.predict(fcpo_testdata)

    fcpo_pred_rescaled_values=robust_scaler_labels.inverse_transform(np.array(fcpo_pred_values).reshape(1,-1))
    fcpo_pred_rescaled_values=pd.Series(fcpo_pred_rescaled_values[0],index=fcpo_testlabels.index)
    fcpo_pred_values=pd.Series(fcpo_pred_values,index=fcpo_testlabels.index)
    mean_absolute_error(fcpo_testlabels,fcpo_pred_values)

    plt.figure(figsize=(10,10))
    fcpo_testlabels.plot()
    pd.Series(fcpo_pred_values,index=fcpo_testlabels.index).plot()
