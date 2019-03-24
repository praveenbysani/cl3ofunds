import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

import random
random.seed(333)
np.random.seed(333)

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

from data_utils import prepare_daily_data, get_first_element, get_last_element,transform_minute_to_daily_df
from data_utils import generate_tech_ind
from data_utils import generate_candlestick_ind
from backtesting_utils import calculate_long_returns
from backtesting_utils import calculate_short_returns
from backtesting_utils import max_drawdown


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

## function to filter out the non-active trading hours from the raw data
def filter_inactive_hours(df,start_time=900,end_time=1530):
    df = df[(df['Time'] > 900) & (df['Time']< 1530)]
    return df

if __name__=='__main__':

    nasdaq_data=pd.read_csv('data/mini_nasdaq.txt')
    dowj_data = pd.read_csv('data/mini_dowj.txt')
    #fcpo_data = pd.read_csv('data/FCPO_2007-2017_backadjusted.csv')

    dowj_data=filter_inactive_hours(dowj_data)
    dowj_daily_data=transform_minute_to_daily_df(csv_file=None,df=dowj_data)



    lookout_config=1 #including current day
    profit_thr_pct=1
    loss_thr_pct=profit_thr_pct+1
    train_start='2005-01-01'
    train_end='2016-12-31'
    test_start='2017-01-01'
    test_end='2019-01-01'
    commission_price=60
    num_units=25
    initial_book_value=10000

    dowj_daily_df=prepare_daily_data(dowj_daily_data,long_spread_thr=profit_thr_pct,
                                        short_spread_thr=profit_thr_pct,lookup_period=lookout_config)

    #dowj_daily_df.groupby(dowj_daily_df.index.year).agg({'long_spread':'median','short_spread':'median'})

    dowj_daily_tind=generate_tech_ind(dowj_daily_df[['Open','High','Low','Close','Volume']].shift(1))
    dowj_daily_cdlind=generate_candlestick_ind(dowj_daily_df[['Open','High','Low','Close']].shift(1))
    dowj_nadjusted_feats=dowj_daily_tind.merge(dowj_daily_cdlind,left_index=True,right_index=True)

    lprofit_rsearch,lprofit_test_proba,lprofit_test_predlabels,lprofit_testdata,lprofit_testlabels=build_best_model(
                                dowj_daily_df,dowj_nadjusted_feats,'lprofit_ind',100,train_start,train_end,test_start,test_end)

    lprofit_test_predlabels=derive_classification_labels(lprofit_test_proba,90)
    #lprofit_rsearch.best_estimator_
    dowj_eval_df=dowj_daily_df[['Open',
                    'next_nhigh','next_nlow','next_nclose',
                    'lprofit_ind','sprofit_ind']].merge(
        pd.DataFrame(lprofit_test_predlabels,columns=['lprofit_prediction'],index=lprofit_testlabels.index),
            left_index=True,right_index=True)
