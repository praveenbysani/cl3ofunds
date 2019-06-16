import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler
import numpy as np
import seaborn as sns
sns.set()

data_orig = pd.read_csv('data/fcpo_daily_2010_2018.csv')
data_orig=data_orig.set_index('Date')
data_orig = data_orig[['Open','High','Low','Close','Volume']]

gmm = GaussianMixture(n_components=4,covariance_type='spherical',n_init=1000,random_state=4)
regime_info=gmm.fit_predict(data_orig)
data_orig=data_orig.assign(regime=regime_info)

rscaler=RobustScaler()
data_sclaed=rscaler.fit_transform(data_orig)

plt.figure(figsize=(20,12))
sns.lineplot(x=data_orig.index,y=data_orig['Close'],hue=data_orig['regime'])
