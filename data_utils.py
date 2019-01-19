import warnings
import pandas as pd
import numpy as np
import os
import random
import talib
from tech_indicators import stoch,stoch_signal

warnings.filterwarnings("ignore")
np.random.seed(33)
random.seed(33)

def zscore_func_improved(x,window_size=20):
    rolling_mean=x.rolling(window=window_size).mean().bfill()
    rolling_std = x.rolling(window=window_size).std().bfill()
    return (x-rolling_mean)

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
    return round((x['next_nhigh']-x['Open'])/x['Open'],4)*100

def compute_short_spread(x):
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
                                long_spread = fcpo_data.apply(lambda x: compute_long_spread(x),axis=1),
                                short_spread = fcpo_data.apply(lambda x: compute_short_spread(x),axis=1)
                            )
    fcpo_data=fcpo_data.assign(long_short_spread_diff= fcpo_data['long_spread']-fcpo_data['short_spread'])
    fcpo_data['prev_open_change_pct']=fcpo_data['prev_open_change_pct'].bfill()
    return fcpo_data


def generate_tech_ind(fcpo_df):
    fcpo_df_tind=pd.DataFrame({
        'ema_slow_close':talib.EMA(fcpo_df['Close'],timeperiod=25).bfill(),
        'ema_fast_close':talib.EMA(fcpo_df['Close'],timeperiod=10).bfill(),
        #'sma_slow_close': talib.SMA(fcpo_df['Close'],timeperiod=25).bfill(),
        #'sma_fast_close': talib.SMA(fcpo_df['Close'],timeperiod=10).bfill(),
        'macd_close': talib.MACD(fcpo_df['Close'])[0].bfill(),
        'macd_signal_close': talib.MACD(fcpo_df['Close'])[1].bfill(),
        'rsi_close':talib.RSI(fcpo_df['Close']).bfill(),
        'stoch' : stoch(fcpo_df['High'],fcpo_df['Low'],fcpo_df['Close']).bfill(),
        'stoch_signal' : stoch_signal(fcpo_df['High'],fcpo_df['Low'],fcpo_df['Close']).bfill(),
        'adx': talib.ADX(fcpo_df['High'],fcpo_df['Low'],fcpo_df['Close']).bfill(),
        'atr': talib.ATR(fcpo_df['High'],fcpo_df['Low'],fcpo_df['Close']).bfill(),
        'cci': talib.CCI(fcpo_df['High'],fcpo_df['Low'],fcpo_df['Close']).bfill()
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
