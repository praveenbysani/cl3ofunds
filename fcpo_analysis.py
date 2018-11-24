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

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RandomizedSearchCV

import xgboost

## functions necessary to group the minute level to hours/days
def get_first_element(x):
    return x.iloc[0]

def get_last_element(x):
    return x.iloc[len(x)-1]

# indicate the indicator as positive when the price is within specified threshold
##ranges compared to the open for the long strategy
def infer_profit_indicator(x,lp_ind='long_spread',threshold=2):
    if x[lp_ind] >= threshold:
        return 1
    else:
        return 0

def compute_long_spread(x):
    return round((max(x['next_1high'],x['next_2high'],x['next_3high'],x['next_4high'],x['High'])-x['Open'])/x['Open'],4)*100

def compute_short_spread(x):
    return round((x['Open']-min(x['next_1low'],x['next_2low'],x['next_3low'],x['next_4low'],x['Low']))/x['Open'],4)*100

def zscore_func_improved(x,window_size=20):
    rolling_mean=x.rolling(window=window_size).mean().bfill()
    rolling_std = x.rolling(window=window_size).std().bfill()
    return (x-rolling_mean)

def prepare_daily_data(fcpo_data,long_spread_thr=2,short_spread_thr=2,lookup_period=5):
    fcpo_data=fcpo_data.assign(
                          next_1open=fcpo_data['Open'].shift(-1),
                          next_1high=fcpo_data['High'].shift(-1),
                          next_1low=fcpo_data['Low'].shift(-1),
                          next_1close=fcpo_data['Close'].shift(-1),
                          next_2open=fcpo_data['Open'].shift(-2),
                          next_2high=fcpo_data['High'].shift(-2),
                          next_2low=fcpo_data['Low'].shift(-2),
                          next_2close=fcpo_data['Close'].shift(-2),
                          next_3open=fcpo_data['Open'].shift(-3),
                          next_3high=fcpo_data['High'].shift(-3),
                          next_3low=fcpo_data['Low'].shift(-3),
                          next_3close=fcpo_data['Close'].shift(-3),
                          next_4open=fcpo_data['Open'].shift(-4),
                          next_4high=fcpo_data['High'].shift(-4),
                          next_4low=fcpo_data['Low'].shift(-4),
                          next_4close=fcpo_data['Close'].shift(-4),

                          prev_1open=fcpo_data['Open'].shift(1),
                          prev_1high=fcpo_data['High'].shift(1),
                          prev_1low=fcpo_data['Low'].shift(1),
                          prev_1close=fcpo_data['Close'].shift(1),
                          )
    ## infer the profit indicators based on the future closing/high or low  prices
    # compute the long-profit and short-profit indicators
    # compute for each day, the open change pct compared to the previoius day
    # compute for each day, the next day open change pct, compared to current day close (only use for evaluation/validation purposes)
    fcpo_data=fcpo_data.assign(
                                prev_open_change_pct=(fcpo_data['Open']/fcpo_data['prev_1close'])*100-100,
                                next_open_change_pct=(fcpo_data['next_1open']/fcpo_data['Close'])*100-100,
                                long_spread = fcpo_data.apply(lambda x: compute_long_spread(x),axis=1),
                                short_spread = fcpo_data.apply(lambda x: compute_short_spread(x),axis=1)
                                )
    fcpo_data=fcpo_data.assign(
                     lprofit_ind=fcpo_data.apply(lambda x: infer_profit_indicator(x,'long_spread',long_spread_thr),axis=1),
                     sprofit_ind=fcpo_data.apply(lambda x: infer_profit_indicator(x,'short_spread',short_spread_thr),axis=1))
    fcpo_data['prev_open_change_pct']=fcpo_data['prev_open_change_pct'].bfill()
    return fcpo_data

def generate_tech_ind(fcpo_df):
    fcpo_df_tind=pd.DataFrame({
        'ema_slow_open':ta.trend.ema_slow(fcpo_df['Open'],n_slow=25).bfill(),
        'ema_fast_open':ta.trend.ema_fast(fcpo_df['Open'],n_fast=10).bfill(),
        'sma_slow_open': talib.SMA(fcpo_df['Open'],timeperiod=25).bfill(),
        'sma_fast_open': talib.SMA(fcpo_df['Open'],timeperiod=10).bfill(),
        'macd_open': ta.trend.macd(fcpo_df['Open']).bfill(),
        'macd_signal_open': ta.trend.macd_signal(fcpo_df['Open']).bfill(),
        'rsi_open':ta.momentum.rsi(fcpo_df['Open']).bfill(),
        'ema_slow_high':ta.trend.ema_slow(fcpo_df['High'],n_slow=25).bfill(),
        'ema_fast_high':ta.trend.ema_fast(fcpo_df['High'],n_fast=10).bfill(),
        'sma_slow_high': talib.SMA(fcpo_df['High'],timeperiod=25).bfill(),
        'sma_fast_high': talib.SMA(fcpo_df['High'],timeperiod=10).bfill(),
        'macd_high': ta.trend.macd(fcpo_df['High']).bfill(),
        'macd_signal_high': ta.trend.macd_signal(fcpo_df['High']).bfill(),
        'rsi_high':ta.momentum.rsi(fcpo_df['High']).bfill(),
        'ema_slow_low':ta.trend.ema_slow(fcpo_df['Low'],n_slow=25).bfill(),
        'ema_fast_low':ta.trend.ema_fast(fcpo_df['Low'],n_fast=10).bfill(),
        'sma_slow_low': talib.SMA(fcpo_df['Low'],timeperiod=25).bfill(),
        'sma_fast_low': talib.SMA(fcpo_df['Low'],timeperiod=10).bfill(),
        'macd_low': ta.trend.macd(fcpo_df['Low']).bfill(),
        'macd_signal_low': ta.trend.macd_signal(fcpo_df['Low']).bfill(),
        'rsi_low':ta.momentum.rsi(fcpo_df['Low']).bfill(),
        'ema_slow_close':ta.trend.ema_slow(fcpo_df['Close'],n_slow=25).bfill(),
        'ema_fast_close':ta.trend.ema_fast(fcpo_df['Close'],n_fast=10).bfill(),
        'sma_slow_close': talib.SMA(fcpo_df['Close'],timeperiod=25).bfill(),
        'sma_fast_close': talib.SMA(fcpo_df['Close'],timeperiod=10).bfill(),
        'macd_close': ta.trend.macd(fcpo_df['Close']).bfill(),
        'macd_signal_close': ta.trend.macd_signal(fcpo_df['Close']).bfill(),
        'rsi_close':ta.momentum.rsi(fcpo_df['Close']).bfill(),
        'stoch' : stoch(fcpo_df['High'],fcpo_df['Low'],fcpo_df['Close']).bfill(),
        'stoch_signal' : stoch_signal(fcpo_df['High'],fcpo_df['Low'],fcpo_df['Close']).bfill(),
        'rsi_close':ta.momentum.rsi(fcpo_df['Close']).bfill(),
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

def build_modeling_data(fcpo_data_daily,fcpo_feats,target_label='lprofit_ind',
                        train_start='2015-01-01',train_end='2017-12-31',test_start='2018-01-01',test_end='2018-10-01'):
    fcpo_train=fcpo_feats[train_start:train_end]
    fcpo_test = fcpo_feats[test_start:test_end]

    fcpo_train=fcpo_train.merge(fcpo_data_daily[['Open','prev_open_change_pct',target_label]],left_index=True,right_index=True)
    fcpo_traindata=fcpo_train.iloc[:,0:-1]
    fcpo_trainlabels=fcpo_train.iloc[:,-1:]

    fcpo_test=fcpo_test.merge(fcpo_data_daily[['Open','prev_open_change_pct',target_label]],left_index=True,right_index=True)
    fcpo_testdata=fcpo_test.iloc[:,0:-1]
    fcpo_testlabels=fcpo_test.iloc[:,-1]
    return fcpo_traindata,fcpo_trainlabels,fcpo_testdata,fcpo_testlabels

def buil_cross_validated_model(ml_model,traindata,trainlabels,tsplit,scoring_metric='roc_auc'):
    ml_cval_scores=cross_val_score(ml_model,traindata,np.ravel(trainlabels),
                                       scoring=scoring_metric,cv=tsplit)
    ml_model.fit(traindata,np.ravel(trainlabels))
    return ml_model,ml_cval_scores

def derive_classification_labels(x,threshold=85):
    clf_thr=np.percentile(x[:,1],q=threshold)
    clf_labels=list(map(lambda x: 1 if x > clf_thr else 0, x[:,1]))
    return clf_labels

def build_best_model(fcpo_daily,fcpo_feats,profit_ind='lprofit_ind',n_iter=250,
                        train_start='2011-01-01',train_end='2016-12-31',test_start='2017-01-01',test_end='2018-01-01'):
    fcpo_traindata,fcpo_trainlabels,fcpo_testdata,fcpo_testlabels=build_modeling_data(
                            fcpo_daily,fcpo_feats,profit_ind,train_start,train_end,test_start,test_end)

    fcpo_tsplit=TimeSeriesSplit(n_splits=3)

    xgb_param_dist={
                "n_estimators": [200,300,500],
                "max_depth": [5,10,20],
                "learning_rate": [0.01,0.05,0.1],
                "gamma": [0,0.01,0.05],
                "subsample": [0.5,0.8,1],
                "colsample_bytree": [0.5,0.8,1]
                }

    xgb_rsearch=RandomizedSearchCV(xgboost.XGBClassifier(random_state=333),
                            xgb_param_dist,random_state=333,cv=fcpo_tsplit,scoring='roc_auc',n_iter=n_iter,n_jobs=-1)

    xgb_rsearch=xgb_rsearch.fit(fcpo_traindata,np.ravel(fcpo_trainlabels))

    xgb_test_proba=xgb_rsearch.best_estimator_.predict_proba(fcpo_testdata)
    xgb_test_predlabels=xgb_rsearch.best_estimator_.predict(fcpo_testdata)
    return xgb_rsearch,xgb_test_proba,xgb_test_predlabels,fcpo_testdata,fcpo_testlabels


def calculate_long_returns(x,profit_field,pred_field,commission_price,num_units,profit_thr=0.03,loss_thr=0.01):
    return_myr=0
    #when the prediction is zero, the model accumulates no returns
    if x[pred_field]==0:
        return_myr=0
    #when the model makes a prediction
    elif x[pred_field]==1:
        #when it is correct, sell it off when it reaches max x% profit zone
        if x[profit_field]==1:
            return_myr=((x['Open']*(1+profit_thr)-x['Open'])*num_units)-commission_price
        #when it went wrong, sell it off when it reaches x% loss or wait till it reaches eod
        elif x[profit_field]==0:
            max_loss_val=min(x['Low'],x['next_1low'],x['next_2low'],x['next_3low'],x['next_4low'])-x['Open']
            thr_loss_val=(x['Open']*(1-loss_thr))-x['Open']
            if max_loss_val < thr_loss_val:
                return_myr=thr_loss_val*num_units-commission_price
            else:
                return_myr= (x['next_4close']-x['Open'])*num_units-commission_price
    return return_myr

def calculate_short_returns(x,profit_field,pred_field,commission_price,num_units,profit_thr=0.03,loss_thr=0.01):
    return_myr=0
    #when the prediction is zero, the model accumulates no returns
    if x[pred_field]==0:
        return_myr=0
    #when the model makes a prediction
    elif x[pred_field]==1:
        if x[profit_field]==1:
        #when it is correct, sell it off when it reaches max x% profit zone
            return_myr=((x['Open']-x['Open']*(1-profit_thr))*num_units)-commission_price
        #when it went wrong, sell it off when it reaches x% loss or wait till it reaches eod
        elif x[profit_field]==0:
            max_loss_val=x['Open']-max(x['High'],x['next_1high'],x['next_2high'],x['next_3high'],x['next_4high'])
            thr_loss_val=x['Open']-(x['Open']*(1+loss_thr))
            if max_loss_val < thr_loss_val:
                return_myr=thr_loss_val*num_units-commission_price
            else:
                return_myr=(x['Open']-x['next_4close'])*num_units-commission_price
    return return_myr

def max_drawdown(X):
    mdd = 0
    peak = X[0]
    mdd_peak=0
    mdd_x=0
    for x in X:
        if x > peak:
            peak = x
        dd = (peak - x) / peak
        if dd > mdd:
            mdd = dd
            mdd_peak=peak
            mdd_x=x
    return mdd_peak,mdd_x,mdd

if __name__=='__main__':
    fcpo_data=pd.read_csv('data/fcpo_daily_2018.csv')
    fcpo_data=fcpo_data[['Date','Open','High','Low','Close','Volume']]
    fcpo_data=fcpo_data.set_index(pd.to_datetime(fcpo_data['Date']))
    fcpo_data=fcpo_data.drop(columns=['Date'])
    # prepare the data with training indicators, specify the profit thresholds
    fcpo_daily_nadjusted=prepare_daily_data(fcpo_data,long_spread_thr=2,short_spread_thr=2)

    fcpo_daily_tind=generate_tech_ind(fcpo_daily_nadjusted[['Open','High','Low','Close','Volume']].shift(1))
    fcpo_daily_cdlind=generate_candlestick_ind(fcpo_daily_nadjusted[['Open','High','Low','Close']].shift(1))
    fcpo_nadjusted_feats=fcpo_daily_tind.merge(fcpo_daily_cdlind,left_index=True,right_index=True)

    # build the hyper parameter tuned xgboost models and specify the training and testing periods
    sprofit_rsearch,sprofit_test_proba,sprofit_test_predlabels,sprofit_testdata,sprofit_testlabels=build_best_model(
                                        fcpo_daily_nadjusted,fcpo_nadjusted_feats,'sprofit_ind',250,'2014-01-01','2017-12-31','2018-01-01','2018-12-31')
    lprofit_rsearch,lprofit_test_proba,lprofit_test_predlabels,lprofit_testdata,lprofit_testlabels=build_best_model(
                                        fcpo_daily_nadjusted,fcpo_nadjusted_feats,'lprofit_ind',250,'2014-01-01','2017-12-31','2018-01-01','2018-12-31')
    print("Short Model:: XGB precision : {pscore}, recall: {rscore} auc: {auc_score}".format(
                            pscore=round(precision_score(sprofit_testlabels,sprofit_test_predlabels),2),
                            rscore=round(recall_score(sprofit_testlabels,sprofit_test_predlabels),2),
                            auc_score=round(roc_auc_score(sprofit_testlabels,sprofit_test_proba[:,1]),2)))
    print("Long Model:: XGB precision : {pscore}, recall: {rscore} auc: {auc_score}".format(
                            pscore=round(precision_score(lprofit_testlabels,lprofit_test_predlabels),2),
                            rscore=round(recall_score(lprofit_testlabels,lprofit_test_predlabels),2),
                            auc_score=round(roc_auc_score(lprofit_testlabels,lprofit_test_proba[:,1]),2)))

    sprofit_rsearch.best_score_
    lprofit_rsearch.best_score_


    fcpo_eval_df=fcpo_daily_nadjusted[['Open',
                    'High','next_1high','next_2high','next_3high','next_4high',
                    'Low','next_1low','next_2low','next_3low','next_4low','next_4close',
                    'lprofit_ind','sprofit_ind']].merge(
        pd.DataFrame(sprofit_test_predlabels,columns=['sprofit_prediction'],index=sprofit_testlabels.index),
                left_index=True,right_index=True).merge(
        pd.DataFrame(lprofit_test_predlabels,columns=['lprofit_prediction'],index=lprofit_testlabels.index),
            left_index=True,right_index=True)

    fcpo_eval_df['sprofit_returns']=fcpo_eval_df.apply(lambda x:
                                calculate_short_returns(x,'sprofit_ind','sprofit_prediction',60,25,0.02,0.03),axis=1)
    fcpo_eval_df['lprofit_returns']=fcpo_eval_df.apply(lambda x:
                            calculate_long_returns(x,'lprofit_ind','lprofit_prediction',60,25,0.02,0.03),axis=1)

    fcpo_eval_df=fcpo_eval_df.assign(cummulative_sprofit_returns=fcpo_eval_df['sprofit_returns'].cumsum(),
                                    cummulative_lprofit_returns=fcpo_eval_df['lprofit_returns'].cumsum())
    fcpo_eval_df=fcpo_eval_df.assign(cummulative_sprofit_returns=fcpo_eval_df['cummulative_sprofit_returns']+10000,
                                    cummulative_lprofit_returns=fcpo_eval_df['cummulative_lprofit_returns']+10000)

    plt.figure(figsize=(15,10))
    fcpo_eval_df['2017-01-01':'2018-10-01']['cummulative_sprofit_returns'].plot()
    fcpo_eval_df['2017-01-01':'2018-10-01']['cummulative_lprofit_returns'].plot()
