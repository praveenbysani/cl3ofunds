# this script will search across
## different future waiting periods from 2-7 days
## different profit thresholds between 1.5,1.75,2,2.25,2.5 percentages
## different training periods, look-back of 3-5 years
# to find the scenario which gives better returns with minimal drawdown
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

from data_utils import prepare_daily_data
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




if __name__=='__main__':
    fcpo_data=pd.read_csv('data/fcpo_daily_2010_2018.csv')
    fcpo_data=fcpo_data[['Date','Open','High','Low','Close','Volume']]
    fcpo_data=fcpo_data.set_index(pd.to_datetime(fcpo_data['Date']))
    fcpo_data=fcpo_data.drop(['Date'],axis=1)

    lookout_config_list=[5,2,7]
    profit_thr_pct_list=[2.5,3]
    train_start_list=['2013-01-01']
    train_end_list=['2017-12-31']
    test_start_list=['2018-01-01']
    test_end_list=['2018-10-01']

    commission_price=60
    num_units=25
    initial_book_value=10000

    ## Confiure all parameters
    lookout_config=7 #including current day
    profit_thr_pct=2.5
    train_start='2013-01-01'
    train_end='2017-12-31'
    test_start='2018-01-01'
    test_end='2018-10-01'
    #plt.figure(figsize=(10,10))
    #plt.plot(fcpo_daily_nadjusted['2012-01-01':'2013-01-01']['Close'])

    ## iterate over all combinations of train-test data split combinations
    for train_start,train_end,test_start,test_end in zip(train_start_list,train_end_list,test_start_list,test_end_list):
        print(train_start,train_end,test_start,test_end)
        train_prefix=train_start.split('-')[0]+train_end.split('-')[0]
        test_prefix=test_start.split('-')[0]
        #iterate over all the different lookup period data
        for lookout_config in lookout_config_list:
            print("finding for look out period:{lookout_period}".format(lookout_period=lookout_config))
            for profit_thr_pct in profit_thr_pct_list:
                print("searching for profit threshold:{profit_thr}".format(profit_thr=profit_thr_pct))

                loss_thr_pct=profit_thr_pct+1
                rsearch_iters=250
                #NOTE:: prepare the data with training indicators, specify the profit thresholds and the lookout period
                start_time=pd.datetime.now()
                fcpo_daily_nadjusted=prepare_daily_data(fcpo_data,long_spread_thr=profit_thr_pct,short_spread_thr=profit_thr_pct,lookup_period=lookout_config)

                fcpo_daily_tind=generate_tech_ind(fcpo_daily_nadjusted[['Open','High','Low','Close','Volume']].shift(1))
                fcpo_daily_cdlind=generate_candlestick_ind(fcpo_daily_nadjusted[['Open','High','Low','Close']].shift(1))
                fcpo_nadjusted_feats=fcpo_daily_tind.merge(fcpo_daily_cdlind,left_index=True,right_index=True)

                #NOTE:: build the hyper parameter tuned xgboost models and specify the training and testing periods
                sprofit_rsearch,sprofit_test_proba,sprofit_test_predlabels,sprofit_testdata,sprofit_testlabels=build_best_model(
                                                    fcpo_daily_nadjusted,fcpo_nadjusted_feats,'sprofit_ind',rsearch_iters,train_start,train_end,test_start,test_end)


                lprofit_rsearch,lprofit_test_proba,lprofit_test_predlabels,lprofit_testdata,lprofit_testlabels=build_best_model(
                                                    fcpo_daily_nadjusted,fcpo_nadjusted_feats,'lprofit_ind',rsearch_iters,train_start,train_end,test_start,test_end)

                fcpo_eval_df=fcpo_daily_nadjusted[['Open',
                                'next_nhigh','next_nlow','next_nclose',
                                'lprofit_ind','sprofit_ind']].merge(
                    pd.DataFrame(sprofit_test_predlabels,columns=['sprofit_prediction'],index=sprofit_testlabels.index),
                            left_index=True,right_index=True).merge(
                    pd.DataFrame(lprofit_test_predlabels,columns=['lprofit_prediction'],index=lprofit_testlabels.index),
                        left_index=True,right_index=True)

                fcpo_eval_df['sprofit_returns']=fcpo_eval_df.apply(lambda x:
                                            calculate_short_returns(x,'sprofit_ind','sprofit_prediction','lprofit_prediction',commission_price,num_units,profit_thr_pct*0.01,loss_thr_pct*0.01),axis=1)
                fcpo_eval_df['lprofit_returns']=fcpo_eval_df.apply(lambda x:
                                        calculate_long_returns(x,'lprofit_ind','lprofit_prediction','sprofit_prediction',commission_price,num_units,profit_thr_pct*0.01,loss_thr_pct*0.01),axis=1)

                fcpo_eval_df=fcpo_eval_df.assign(cummulative_sprofit_returns=fcpo_eval_df['sprofit_returns'].cumsum(),
                                                cummulative_lprofit_returns=fcpo_eval_df['lprofit_returns'].cumsum())
                fcpo_eval_df=fcpo_eval_df.assign(cummulative_sprofit_returns=fcpo_eval_df['cummulative_sprofit_returns']+initial_book_value,
                                                cummulative_lprofit_returns=fcpo_eval_df['cummulative_lprofit_returns']+initial_book_value)
                sprofit_mdd=max_drawdown(fcpo_eval_df['cummulative_sprofit_returns'])
                lprofit_mdd=max_drawdown(fcpo_eval_df['cummulative_lprofit_returns'])

                end_time=pd.datetime.now()
                model_duration=(end_time-start_time).total_seconds()

                ##Export model file and the output predictions for 2017 and 2018
                from sklearn.externals import joblib
                joblib.dump([lprofit_rsearch,sprofit_rsearch,fcpo_eval_df],
                                    '{testprefix}_tr{trainprefix}_{lookout}day_{profit_thr}pct.joblib'.format(testprefix=test_prefix,trainprefix=train_prefix,lookout=lookout_config,profit_thr=profit_thr_pct))

                with open('output/model_results_{lookout}day_{profit}pct.txt'.format(lookout=lookout_config,profit=profit_thr_pct),'a') as result_file:
                    result_file.write("\n################### model duration(secs):{duration} ################################## \n".format(duration=model_duration))
                    result_file.write("train_start:{trstart}, train_end:{trend}, test_start:{tstart}, test_end:{tend}\n".format(
                                        trstart=train_start,trend=train_end,tstart=test_start,tend=test_end))
                    result_file.write("\nShort Model CV AUC::{best_score}\n".format(best_score=round(sprofit_rsearch.best_score_,2)))
                    result_file.write("Short Model:: XGB precision : {pscore}, recall: {rscore} auc: {auc_score}\n".format(
                                            pscore=round(precision_score(sprofit_testlabels,sprofit_test_predlabels),2),
                                            rscore=round(recall_score(sprofit_testlabels,sprofit_test_predlabels),2),
                                            auc_score=round(roc_auc_score(sprofit_testlabels,sprofit_test_proba[:,1]),2)))
                    result_file.write("Short Model Final Return:{total_return}, and Draw Down::{from_value},{to_value},{pct_value}\n\n".format(
                                total_return=fcpo_eval_df['cummulative_sprofit_returns'].iloc[-1:][0],from_value=sprofit_mdd[0],to_value=sprofit_mdd[1],pct_value=sprofit_mdd[2]))

                    result_file.write("\nLong Model CV AUC::{best_score}\n".format(best_score=round(lprofit_rsearch.best_score_,2)))
                    result_file.write("Long Model:: XGB precision : {pscore}, recall: {rscore} auc: {auc_score}\n".format(
                                            pscore=round(precision_score(lprofit_testlabels,lprofit_test_predlabels),2),
                                            rscore=round(recall_score(lprofit_testlabels,lprofit_test_predlabels),2),
                                            auc_score=round(roc_auc_score(lprofit_testlabels,lprofit_test_proba[:,1]),2)))
                    result_file.write("Long Model Final Return:{total_return}, and Draw Down::{from_value},{to_value},{pct_value}\n".format(
                                total_return=fcpo_eval_df['cummulative_lprofit_returns'].iloc[-1:][0],from_value=lprofit_mdd[0],to_value=lprofit_mdd[1],pct_value=lprofit_mdd[2]))

                plt.figure(figsize=(15,10))
                fcpo_eval_df['2017-01-01':'2018-10-01']['cummulative_sprofit_returns'].plot()
                fcpo_eval_df['2017-01-01':'2018-10-01']['cummulative_lprofit_returns'].plot()
                plt.legend(['short_model','long_model'])
                plt.savefig('output/returns_{train_prefix}_{test_prefix}_{lookout}day_{profit}pct_{iters}.jpg'.format(
                                train_prefix=train_prefix,test_prefix=test_prefix,lookout=lookout_config,profit=profit_thr_pct,iters=rsearch_iters))
