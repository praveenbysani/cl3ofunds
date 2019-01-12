import pandas as pd
import numpy as np
import random
import tpot
from sklearn.model_selection import TimeSeriesSplit

random.seed(999)
np.random.seed(999)

def build_modeling_data(fcpo_data_daily,fcpo_feats,target_label='lprofit_ind',split_date='2017-01-01'):
    fcpo_train=fcpo_feats[:split_date]
    fcpo_test = fcpo_feats[split_date:]

    fcpo_train=fcpo_train.merge(fcpo_data_daily[['Open','prev_open_change_pct',target_label]],left_index=True,right_index=True)
    fcpo_traindata=fcpo_train.iloc[:,0:-1]
    fcpo_trainlabels=fcpo_train.iloc[:,-1:]

    fcpo_test=fcpo_test.merge(fcpo_data_daily[['Open','prev_open_change_pct',target_label]],left_index=True,right_index=True)
    fcpo_testdata=fcpo_test.iloc[:,0:-1]
    fcpo_testlabels=fcpo_test.iloc[:,-1]
    return fcpo_traindata,fcpo_trainlabels,fcpo_testdata,fcpo_testlabels


fcpo_daily_nadjusted=pd.read_hdf('data/processed_dta.h5','fcpo_daily_nadjusted')
fcpo_nadjusted_tind=pd.read_hdf('data/processed_dta.h5','fcpo_nadjusted_tind')
fcpo_nadjusted_cdlind=pd.read_hdf('data/processed_dta.h5','fcpo_nadjusted_cdlind')

fcpo_nadjusted_feats=fcpo_nadjusted_tind.merge(fcpo_nadjusted_cdlind,left_index=True,right_index=True)
fcpo_nadjusted_feats=fcpo_nadjusted_feats['2010-01-01':'2018-10-01']

fcpo_tsplit=TimeSeriesSplit(n_splits=3)


fcpo_lprofit_traindata,fcpo_lprofit_trainlabels,fcpo_lprofit_testdata,fcpo_lprofit_testlabels=build_modeling_data(
                                            fcpo_daily_nadjusted,fcpo_nadjusted_feats,'lprofit_ind','2017-01-01')

fcpo_sprofit_traindata,fcpo_sprofit_trainlabels,fcpo_sprofit_testdata,fcpo_sprofit_testlabels=build_modeling_data(fcpo_daily_nadjusted,
                                                                fcpo_nadjusted_feats,'sprofit_ind','2017-01-01')

tpot_lclf=tpot.TPOTClassifier(warm_start=True,verbosity=3,random_state=333,cv=fcpo_tsplit,scoring='roc_auc',
                             generations=5,population_size=30,offspring_size=30,max_time_mins=5)

tpot_lclf.fit(fcpo_lprofit_traindata,fcpo_lprofit_trainlabels)



tpot_sclf=tpot.TPOTClassifier(warm_start=True,verbosity=2,random_state=333,cv=fcpo_tsplit,scoring='roc_auc',
                             generations=5,population_size=30,offspring_size=30)
