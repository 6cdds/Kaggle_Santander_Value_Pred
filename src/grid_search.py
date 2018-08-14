# -*- coding: utf-8 -*-

import pickle
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestRegressor

def random_grid_search():

     f = open('C:\Users\Colleen\Documents\Kaggle_Santander_Value_Pred\data\pca_train_data.csv', 'r')
     pca_train_data = pd.read_csv(f).iloc[:,1:]
     f.close()
     
     f = open('C:\Users\Colleen\Documents\Kaggle_Santander_Value_Pred\data\pca_y_train_data.csv', 'r')
     y = pd.read_csv(f)['target']
     f.close()
     

     max_features = ['auto', 'sqrt']
     
     max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
     max_depth.append(None)
     
     min_samples_split = [2, 5, 10]
     
     min_samples_leaf = [1, 2, 4]
     
     bootstrap = [True, False]

     random_grid = {'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
     

     rf = RandomForestRegressor(n_estimators = 200, n_jobs = -1)

     rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, 
                                   n_iter = 100, cv = 3, verbose=2,
                                   random_state=42, n_jobs = -1)
     
     rf_random.fit(pca_train_data, y)
     
     f = open('C:\Users\Colleen\Documents\Kaggle_Santander_Value_Pred\data\rf_random.p', 'w')
     pickle.dump(rf_random, f)
     f.close()

     '''
     grid_search = GridSearchCV(estimator = rf, param_grid = random_grid, 
                               cv = 3, n_jobs = -1, verbose = 2,
                               return_train_score = True,
                               scoring = 'neg_mean_squared_error')
     
     #rand_mat = np.random.random(pca_train_data.shape)
     t0 = time.time()
     grid_search.fit(pca_train_data, train_data['target'])
     t1 = time.time()
     print t1 - t0

     
     f = open('grid_search_res2.p', 'w')
     pickle.dump(grid_search, f)
     f.close()
     '''
     

if __name__ == '__main__':
     random_grid_search()