# -*- coding: utf-8 -*-

from sklearn.ensemble import RandomForestRegressor

import pandas as pd
import cPickle as pickle

def run_rf():
     
     #f = open('C:\Users\Colleen\Documents\Kaggle_Santander_Value_Pred\data\pca_train_data.csv', 'r')
     #f = open('C:\Users\Colleen\Documents\Kaggle_Santander_Value_Pred\scale_train_data.csv', 'r')
     f = open('C:\\Users\\Colleen\\Documents\\Kaggle_Santander_Value_Pred\\new_scale_train_data.csv', 'r')
     pca_train_data = pd.read_csv(f).iloc[:,1:]
     f.close()    
     

     
     f = open('C:\Users\Colleen\Documents\Kaggle_Santander_Value_Pred\data\pca_y_train_data.csv', 'r')
     y = pd.read_csv(f)['target']
     f.close()

     f = open('C:\Users\Colleen\Documents\Kaggle_Santander_Value_Pred\sel_feats.p', 'r')
     sel_feats = pickle.load(f)
     f.close()

     print pca_train_data.columns
     
     rf = RandomForestRegressor(n_estimators = 1000, max_features = 'auto', 
                                min_samples_leaf = 5, verbose = 2,
                                n_jobs = -1)
     rf.fit(pca_train_data.loc[:, map(str, sel_feats)], y)
     #rf.fit(pca_train_data, y)
     
     f = open('C:\\Users\\Colleen\\Documents\\Kaggle_Santander_Value_Pred\\data\\rf_sel_feats.p', 'w')
     pickle.dump(rf, f)
     f.close()
     
if __name__ == '__main__':
     run_rf()