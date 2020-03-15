# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 09:40:02 2020

@author: Dell
"""
""
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib','qt')
 
import pandas as pd
pd.set_option('display.max_columns',200)
pd.set_option('display.max_rows',500)

import numpy as np
import matplotlib.pyplot as plt

import seaborn as sn

dataset = pd.read_csv('new_train_data.csv')
X = dataset.drop(columns='SalePrice') #independent fields
y= dataset['SalePrice'] #label

#do split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.2,
                                                    random_state = 0)

#do scale not needed since it handeled in python library

#build model
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=50 , random_state = 0)
regressor.fit(X_train,y_train)
##test model
y_predict=regressor.predict(X_test)
## evaluate model
print("Test Data Score: %0.4f" % regressor.score(X_test,y_test))

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(estimator = regressor,
                             X = X_train,
                             y = y_train, cv = 10)
print("Scores: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))

## do parameter tuining
from sklearn.model_selection import GridSearchCV
parameters=[{"n_estimators": [50,100,200],
              "min_samples_leaf": [1, 3, 5],
              'min_samples_split': [2, 5, 10],
              "max_features": ['auto', 'sqrt', 'log2']}]
gridsearch=GridSearchCV(estimator = regressor,
                        param_grid = parameters,
                        scoring = 'r2',
                        cv = 10,
                        n_jobs = -1)
gridsearch = gridsearch.fit(X_train, y_train)
best_score = gridsearch.best_score_
best_parameters = gridsearch.best_params_
# give best parameter as i already select so give same score as K-Fold cross validation