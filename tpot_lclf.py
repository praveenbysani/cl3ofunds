import numpy as np
import pandas as pd
from sklearn.decomposition import FastICA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=333)

# Average CV score on the training set was:0.5944495038650065
exported_pipeline = make_pipeline(
    FastICA(tol=0.2),
    SelectPercentile(score_func=f_classif, percentile=37),
    RandomForestClassifier(bootstrap=True, criterion="entropy", max_features=0.1, min_samples_leaf=9, min_samples_split=20, n_estimators=100)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
