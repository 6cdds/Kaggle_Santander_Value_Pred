# -*- coding: utf-8 -*-

import pickle
import time
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestRegressor

def grid_search():

     f = open('../data/proc_train_data.p', 'r')
     pca_train_data = pickle.load(f)
     f.close()
     
     f = open('../data/train_data.p', 'r')
     train_data = pickle.load(f)
     f.close()
     
     n_estimators = [200, 400, 600, 800] #[int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
     
     max_features = ['auto', 'sqrt']
     
     max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
     max_depth.append(None)
     
     min_samples_split = [2, 5, 10]
     
     min_samples_leaf = [5, 10, 50, 100]
     
     bootstrap = [True, False]
     
     random_grid = {'max_features': max_features,
                    'min_samples_leaf': min_samples_leaf}
     
     #random_grid = {'n_estimators': [n_estimators[-1]]}
     
     
     
     rf = RandomForestRegressor(n_estimators = 500, max_features = 'auto', min_samples_leaf = 1,
                              n_jobs = -1)
      
     '''
     df = pca_train_data.iloc[range(pca_train_data.shape[0] * 2 / 3),:]
     y = train_data['target'].iloc[range(pca_train_data.shape[0] * 2 / 3)]
     
     #df = pca_train_data.iloc[range(2000),:]
     #df = np.random.random(df.shape)
     #y = train_data['target'].iloc[range(2000)]
     
     t0 = time.time()
     rf.fit(df, y)
     t1 = time.time()
     print t1 - t0
     '''
     
     
     
     #rf = RandomForestRegressor()
     #rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, 
     #                               n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
     

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
     

if __name__ == '__main__':
     grid_search()