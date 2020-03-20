import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from tpot.builtins import StackingEstimator

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=42)

# Score on the training set was:0.9431657264943633
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=LogisticRegression(C=0.5, dual=False, penalty="l2")),
    StackingEstimator(estimator=DecisionTreeClassifier(criterion="gini", max_depth=9, min_samples_leaf=6, min_samples_split=7)),
    StackingEstimator(estimator=DecisionTreeClassifier(criterion="entropy", max_depth=10, min_samples_leaf=11, min_samples_split=17)),
    MinMaxScaler(),
    StackingEstimator(estimator=MultinomialNB(alpha=100.0, fit_prior=False)),
    StackingEstimator(estimator=BernoulliNB(alpha=0.01, fit_prior=True)),
    BernoulliNB(alpha=1.0, fit_prior=True)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
