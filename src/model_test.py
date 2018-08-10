# -*- coding: utf-8 -*-

import pickle
import time
import numpy as np
from sklearn.ensemble import RandomForestRegressor

def test_model():
     f = open('../data/proc_train_data.p', 'r')
     proc_train_data = pickle.load(f)
     f.close()
     
     f = open('../data/train_data.p', 'r')
     train_data = pickle.load(f)
     f.close()
     
     f = open('../data/proc_valid_data.p', 'r')
     proc_valid_data = pickle.load(f)
     f.close()
     
     f = open('../data/valid_data.p', 'r')
     valid_data = pickle.load(f)
     f.close()
     
     f = open('../data/grid_search_res2.p', 'r')
     rf_gs_res = pickle.load(f)
     f.close()     
     
     rf = RandomForestRegressor(n_estimators = 500, max_features = 'auto', min_samples_leaf = 1,
                              n_jobs = -1, verbose = 2)
     
     rf.set_params(**rf_gs_res.best_params_)
     
     rf.fit(proc_train_data, train_data['target'])
     
     valid_res = rf.predict(proc_valid_data)
     
     return_res = {'model': rf, 'valid_res': valid_res}
     
     f = open('../data/valid_res_rf.p', 'w')
     pickle.dump(return_res, f)
     f.close()
     
if __name__ == '__main__':
     test_model()