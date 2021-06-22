import dask.dataframe as dd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

os.chdir('/home/lpa263/Competition/Datasets/test4')


df = dd.read_csv('test4_nosmoke*.csv')
df = df.compute()
X = df.iloc[:, :-1]
y = df.iloc[:, -1:]
X = X.to_numpy()
y = y.to_numpy().reshape(-1)
del df
test = pd.read_csv('real_test4_nosmoke.csv')
test = test.iloc[:,:-2].to_numpy()

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
param_grid = {'C' : [.001, .005, .01, .05, .1, .5, 1, 5, 10, 50, 100]}
grid_search = GridSearchCV(LogisticRegression(max_iter = 10000, n_jobs = -1), param_grid, cv = 10, scoring = 'f1', verbose = 2 )
grid_search.fit(X, y)
result = pd.DataFrame(grid_search.cv_results_)
result.to_csv('/home/lpa263/Competition/Datasets/test4/logreg_nosmoke_result.csv', index = False)
predict = pd.DataFrame(grid_search.predict(test), columns = ['target'])
predict.to_csv('/home/lpa263/Competition/Datasets/test4/logreg_nosmoke_prediction.csv', index = False)