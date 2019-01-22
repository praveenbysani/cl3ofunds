import pandas as pd
import numpy as np
import random
random.seed(333)
np.random.seed(333)

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

from importlib import reload
import data_utils
reload(data_utils)
from data_utils import prepare_daily_data
from data_utils import generate_tech_ind
from data_utils import generate_candlestick_ind

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
    fcpo_data=pd.read_csv('data/fcpo_daily_2010_2018.csv')
    fcpo_data=fcpo_data[['Date','Open','High','Low','Close','Volume']]
    fcpo_data=fcpo_data.set_index(pd.to_datetime(fcpo_data['Date']))
    fcpo_data=fcpo_data.drop(['Date'],axis=1)

    fcpo_data_daily=prepare_daily_data(fcpo_data,lookup_period=5)
    fcpo_daily_tind=generate_tech_ind(fcpo_data_daily[['Open','High','Low','Close','Volume']].shift(1))
    fcpo_daily_cdlind=generate_candlestick_ind(fcpo_data_daily[['Open','High','Low','Close']].shift(1))
    fcpo_feats=fcpo_daily_tind.merge(fcpo_daily_cdlind,left_index=True,right_index=True)

    fcpo_traindata,fcpo_trainlabels,fcpo_testdata,fcpo_testlabels=build_regression_model_data(
                            fcpo_data_daily,fcpo_feats,'long_short_spread_diff')


    plt.figure(figsize=(10,10))
    plt.plot(fcpo_data_daily.loc['2017-03-01':'2018-01-01','Close'])


    plt.figure(figsize=(10,10))
    plt.plot(fcpo_data_daily.loc['2017-03-01':'2018-01-01','Close'].rolling(15).std().bfill())



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
