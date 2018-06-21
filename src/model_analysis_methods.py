# -*- coding: utf-8 -*-

""" This module contains general methods to analyse performance of models.
Engine-specific analysis functions are located in the folders for each
engine. (E.g. WebAuth, Tap, Swipe etc...)
"""

import numpy as np
import pandas as pd
import os
import pickle
import copy
from scipy import interp
import itertools
import random

import plotly.graph_objs as go

import utility_methods as um

from sklearn.base import clone
from sklearn.metrics import roc_curve, auc

def featobj_model_learning_curve_preprocessed(
        estimator, preprocessed_data, params, all_feats, unames, 
        user_fold_key, att_fold_key, train_size_key, train_data_key, 
        valid_data_key, train_samps_key, valid_samps_key, 
        scoring_func, save_dir, save_to_file, user_targets, file_add = '',
        test_data_key = '', test_samps_key = ''):
    
    """ Uses preprocessed feature data to generate learning curve data.
    
    
    Args:
        estimator: A sklearn BaseEstimator object with fit and predict calls
        preprocessed_data: A dictionary of user processed features for use
            with cross validation.  The format is as follows:
                {'User1': [{
                        user_fold_key: An integer indicating the user fold
                        att_fold_key: An integer indicating the attacker fold
                        train_samps_key: A list of sample numbers (from 
                            original feature set containing all users) to 
                            be used in training for this fold,
                        test_samps_key: A list of sample numbers (from original
                            feature set containing all users) to be used in 
                            testing for this fold,
                        train_data_key: A dataframe or list of dicts of 
                            training features for this fold,
                        test_data_key: A dataframe or list of dicts of testing 
                            features for this fold},
                        'train_size': The number of samples in the training set
                            {}, ...],
                
                'User2': [...]}
        params: A dictionary containing the model params for each user
        all_feats: A dictionary containing all raw feature objects for
            all users.  It must contain the following columns:
                'feat_group'
                'feat_name'
                'feat_samp_num'
                'feat_time'
                'feat_type'
                'feat_value'
                'user' 
        unames: A list of the user names to use.
        user_fold_key: A string indicating the user fold key in processed_data.
        att_fold_key: A string indicating the attacker fold key in 
            processed_data.
        train_data_key: A string indicating the training data key in 
            processed_data.
        test_data_key: : A string indicating the testing data key in 
            processed_data.
        train_samps_key: A string indicating the training sample numbers key 
            in processed_data.
        test_samps_key: A string indicating the testing sample numbers key in 
            processed_data.
        scorer: A function in the form fun(ground_truth_labels, 
           predicted_labels, uar_save, aar_save) where:
            
            ground_truth_labels: A list mapping the testing samples to actual 
                classes (-1 for not user, 1 for user)
            predict_labels: A list of predicted classes for the testing samples.
            uar_save: An empty list to save the UAR
            aar_save: An empty list to save the AAR
            
        save_dir: A directory where the results can be saved
        file_add: A string containing info to add to file name
        save_to_file: A boolean indicating if the function returns the 
            results or saves to file.                              
    
    Returns:
        If save_to_file is True, it returns nothing and saves a file of
        form '/save_dir/learning_curve_res_u.p' where  u is the user, 
        for each user.
        
        If save_to_file is False, this function returns the results.
        
        The results are in the form:
            {'User1':[{
                    'user_fold': An int indicating the user fold,
                    'att_fold': An int indicating the attacker fold,
                    'train_size': The number of samples used for training in this fold,
                    'train_uar': The training UAR (i.e. The training score),
                    'val_uar': The validation UAR
                    'val_aar': The validation AAR
                    'train_pred': The predicted class labels for the training samples
                    'val_pred': The predicted class labels for the validation samples
                    'val_score': The overall validation score (output of scoring_func)},
                    {}, ....],
            'User2': [...],
            ...}
            
    """
    
    unfit_est = clone(estimator)
    all_res = {}
    for u in unames:
        print u
        lcurve_res = []
        
        # Get preprocessed data info for user
        info = pd.DataFrame([dict((k, x[k]) for k in (user_fold_key, att_fold_key, 
                                  train_size_key, train_data_key, valid_data_key,
                                  test_data_key, train_samps_key, valid_samps_key,
                                  test_samps_key)) \
                                for x in preprocessed_data[u]])
        
        info[user_fold_key] = info[user_fold_key].astype('int')
        info[att_fold_key] = info[att_fold_key].astype('int')
        info[train_size_key] = info[train_size_key].astype('int')
    
        # Get targets
        targets = user_targets[u]
        '''
        targets = []
        for __,row in all_feats.iterrows():
            if row['user'] == u:
                tar = 1
            else:
                tar = -1
            targets.append((row['feat_samp_num'], tar))
        '''

        lcurve_res = []    
        for t_s, g_t_s in info.groupby(train_size_key):
            #print t_s
    
            for user_fold, g_user_fold in g_t_s.groupby(user_fold_key):
                
                est = clone(unfit_est)
                res = est.set_params(**params[u])
                
                # Train model
                train_data = parse_features_data(
                    g_user_fold[train_data_key].iloc[0])

                #train_data = g_user_fold[train_data_key].iloc[0]         
                est.fit(train_data)
    
                # Predict on model - train data
                train_pred = est.predict(train_data)
                #print train_pred
 
                train_uar = []
                train_score = scoring_func(
                        targets, 
                        zip(g_user_fold[train_samps_key].iloc[0], train_pred),
                        uar_save = train_uar)
                train_extra_info = est.get_pred_extra_info()
                
                for att_fold, g_att_fold in g_user_fold.groupby(att_fold_key):
                                        
                    # Predict on model - validation data
                    test_samps = g_att_fold[valid_samps_key].iloc[0]
                
                    test_data = parse_features_data(
                        g_att_fold[valid_data_key].iloc[0],
                        test_samps)

#                    test_data = pd.DataFrame([test_data[k][0] for k in\
#                                              g_att_fold[test_samps_key].iloc[0]])
                    val_pred = est.predict(test_data)

                    val_uar = []
                    val_aar = []
                    val_score = scoring_func(
                            targets, 
                            zip(g_att_fold[valid_samps_key].iloc[0], val_pred),
                            uar_save = val_uar, aar_save = val_aar)
                    extra_info = est.get_pred_extra_info()
                    
                    # Predict on model - test data
                    test_samps = g_att_fold[test_samps_key].iloc[0]
                
                    test_data = parse_features_data(
                        g_att_fold[test_data_key].iloc[0],
                        test_samps)

                    test_pred = est.predict(test_data)

                    test_uar = []
                    test_aar = []
                    test_score = scoring_func(
                            targets, 
                            zip(g_att_fold[test_samps_key].iloc[0], test_pred),
                            uar_save = test_uar, aar_save = test_aar)
                    test_extra_info = est.get_pred_extra_info()
                    
                                
                    lcurve_res.append({'train_size': t_s,
                                       'user_fold': user_fold,
                                       'att_fold': att_fold,
                                       
                                       'train_score': train_score,
                                       'train_pred': train_pred,
                                       'train_uar': train_uar[0],
                                       'train_extra_info': train_extra_info,
                                       
                                       'val_score': val_score,
                                       'val_uar': val_uar[0],
                                       'val_aar': val_aar[0],
                                       'val_pred': val_pred,
                                       'extra_info': extra_info,
                                       
                                       
                                       'test_score': test_score,
                                       'test_uar': test_uar[0],
                                       'test_aar': test_aar[0],
                                       'test_pred': test_pred,
                                       'test_extra_info': test_extra_info
                                       })
    
        if save_to_file:
            f = open(os.path.join(save_dir, 'learning_curve_' + file_add + '_' + u + '.p'), 'w')
            pickle.dump(lcurve_res, f)
            f.close() 
        else:
            all_res[u] = lcurve_res
            
    if not save_to_file:
        return all_res
    
def featobj_model_roc_curve_preprocessed(
        estimator, preprocessed_data, params, all_feats, unames, 
        user_fold_key, att_fold_key, train_data_key, 
        test_data_key, train_samps_key, test_samps_key, 
        save_dir, save_to_file, threshs = None):
    """ Uses preprocessed feature data to generate ROC curve data.
    
    
    Args:
        estimator: A sklearn BaseEstimator object with fit and predict calls
        preprocessed_data: A dictionary of user processed features for use
            with cross validation.  The format is as follows:
                {'User1': [{
                        user_fold_key: An integer indicating the user fold
                        att_fold_key: An integer indicating the attacker fold
                        train_samps_key: A list of sample numbers (from 
                            original feature set containing all users) to 
                            be used in training for this fold,
                        test_samps_key: A list of sample numbers (from original
                            feature set containing all users) to be used in 
                            testing for this fold,
                        train_data_key: A dataframe or list of dicts of 
                            training features for this fold,
                        test_data_key: A dataframe or list of dicts of testing 
                            features for this fold},
                            {}, ...],
                
                'User2': [...]}
        params: A dictionary containing the model params for each user.
        all_feats: A dictionary containing all raw feature objects for
            all users.  It must contain the following columns:
                'feat_group'
                'feat_name'
                'feat_samp_num'
                'feat_time'
                'feat_type'
                'feat_value'
                'user' 
        unames: A list of the user names to use.
        user_fold_key: A string indicating the user fold key in processed_data.
        att_fold_key: A string indicating the attacker fold key in 
            processed_data.
        train_data_key: A string indicating the training data key in 
            processed_data.
        test_data_key: : A string indicating the testing data key in 
            processed_data.
        train_samps_key: A string indicating the training sample numbers key 
            in processed_data.
        test_samps_key: A string indicating the testing sample numbers key in 
            processed_data.
        scorer: A function in the form fun(ground_truth_labels, 
           predicted_labels, uar_save, aar_save) where:
            
            ground_truth_labels: A list mapping the testing samples to actual 
                classes (-1 for not user, 1 for user)
            predict_labels: A list of predicted classes for the testing samples.
            uar_save: An empty list to save the UAR
            aar_save: An empty list to save the AAR
            
        save_dir: A directory where the results can be saved
        file_add: A string containing info to add to file name
        save_to_file: A boolean indicating if the function returns the 
            results or saves to file.    
        threshs: An optional list of score thresholds to use in calculating
            the ROC curves.  If equal to None, the thresholds are generated
            according to sklearn's roc_curve function .                         
    
    Returns:
        If save_to_file is True, it returns nothing and saves a file of
        form '/save_dir/learning_curve_res_u.p' where  u is the user, 
        for each user.
        
        If save_to_file is False, this function returns the results.
        
        The results are in the form:
            {'User1':
                {'mean':{                        
                        'mean_tpr': The fold mean true pos rate curve,
                        'mean_fpr': The fold mean false pos rate curve,
                        'pos_std_tpr': The fold mean TPR curve + std
                        'neg_std_tpr': The fold mean TPR curve - std
                        'mean_auc': The fold mean AUC (calculated using FPR vs TPR)},  
                        'std_auc': The fold std of AUC (calculated using FPR vs TPR)},                                            
                        
                        'mean_uar': The fold mean UAR curve,
                        'mean_aar': The fold mean AAR curve,
                        'pos_std_uar': The fold mean UAR curve + std,
                        'neg_std_uar': The fold mean UAR curve - std,
                        'mean_uar_aar_auc': The fold mean AUC (calculated using UAR vs AAR),
                        'std_uar_aar_auc': The fold std of AUC (calculated using UAR vs AAR)}

                'folds': [{
                        'threshs': The list of thresholds used to calculate the ROC curve
                        'user_fold': An integer indicating the user fold,
                        'att_fold': An integer indicating the attacker fold,

                        'tpr': The true pos rate curve of this fold,
                        'fpr': The false pos rate curve of this fold,
                        'auc': The AUC for this fold, calculated with FPR vs TPR,
                        
                        'uar': The uar curve of this fold,
                        'aar: The aar curve of this fold,
                        'uar_aar_auc': The AUC for this fold, calculated with UAR vs AAR,
                        
                        'train_pred': The scores of the training samples predicted against
                            the model,
                        'val_pred': The scores of the validation samples predicted against
                            the model},
                        {}, ....]},
            'User2': {},
            ...}
            
    """
        
    unfit_est = clone(estimator)
    all_res = {}
    for u in unames:
        print u
        
        # Get preprocessed data info for user
        info = pd.DataFrame([dict((k, x[k]) for k in (user_fold_key, att_fold_key, 
                                  train_data_key, test_data_key,
                                  train_samps_key, test_samps_key)) \
                                for x in preprocessed_data[u]])
        
        info[user_fold_key] = info[user_fold_key].astype('int')
        info[att_fold_key] = info[att_fold_key].astype('int')
    
        # Get targets
        targets = featobj_oneclass_targets(all_feats, u, 
                                           'user', 'feat_samp_num')
        targets_df = pd.DataFrame(targets, columns = ['samp_num', 'tar'])
    
        roccurve_res = []    

        for user_fold, g_user_fold in info.groupby(user_fold_key):
            #print user_fold
            
            est = clone(unfit_est)
            res = est.set_params(**params[u])
            
            # Train model
            train_data = parse_features_data(
                    g_user_fold[train_data_key].iloc[0])

            est.fit(train_data)
            train_pred = est.predict(train_data)
            
            for att_fold, g_att_fold in g_user_fold.groupby(att_fold_key):
                #print att_fold
                                    
                # Predict on model - test data
                test_samps = g_att_fold[test_samps_key].iloc[0]
                
                test_data = parse_features_data(
                    g_att_fold[test_data_key].iloc[0],
                    test_samps)
                    
                val_pred = est.predict(test_data)
                
                y_test = [int(targets_df.loc[targets_df['samp_num'] == x, 
                                             'tar']) for x in test_samps]
                
                # Get traditional roc curve
                fpr, tpr, thresholds = roc_curve(y_test, val_pred)
                
                if threshs != None:
                    thresholds = threshs
                
                # Get UAR/AAR
                uar = []
                aar = []
                for t in thresholds:
                    gt_targ = np.array([x > 0 for x in y_test])
                    
                    if est.bigger_score_better:
                        pred_targ = np.array([x >= t for x in val_pred])
                    else:
                        pred_targ = np.array([x <= t for x in val_pred])
                    uar.append(float(len(pred_targ[(gt_targ) & (pred_targ)])) / float(len(pred_targ[gt_targ])))
                    aar.append(float(len(pred_targ[(np.invert(gt_targ)) & (pred_targ)])) / float(len(pred_targ[np.invert(gt_targ)])))                
                    
                roccurve_res.append({
                        'user_fold': user_fold,
                        'att_fold': att_fold,
                        'fpr': fpr + [0.0],
                        'tpr': tpr + [0.0],
                        'threshs': list(thresholds) + [0.0],
                        'uar': uar + [0.0],
                        'aar': aar + [0.0],
                        'auc': auc(fpr, tpr),
                        'uar_aar_auc': auc(aar, uar),
                        'val_pred': val_pred,
                        'train_pred': train_pred})
    
        # Get mean roc results

        mean_fpr = np.linspace(0, 1, 100)            
        interp_tpr = []
        interp_uar = []
        for i in range(len(roccurve_res)):
            
            # Sort the results
            sort_inds = np.argsort(roccurve_res[i]['fpr'])            
            fold_fpr = np.array(roccurve_res[i]['fpr'])[sort_inds]
            fold_tpr = np.array(roccurve_res[i]['tpr'])[sort_inds]
            
            sort_inds = np.argsort(roccurve_res[i]['aar'])            
            fold_aar = np.array(roccurve_res[i]['aar'])[sort_inds]
            fold_uar = np.array(roccurve_res[i]['uar'])[sort_inds]
            
            interp_tpr.append(interp(mean_fpr, fold_fpr, fold_tpr))
            #interp_tpr[-1][0] = 0.0
            
            interp_uar.append(interp(mean_fpr, fold_aar, fold_uar))
            #interp_uar[-1][0] = 0.0
    
        tpr_df = pd.DataFrame(interp_tpr)
        uar_df = pd.DataFrame(interp_uar)
        mean_roccurve_res = {'mean_fpr': mean_fpr, 
                    'mean_tpr': tpr_df.mean(),
                    'pos_std_tpr': tpr_df.mean() + tpr_df.std(),
                    'neg_std_tpr': tpr_df.mean() - tpr_df.std(),
                    'mean_auc': np.mean([x['auc'] for x in roccurve_res]),
                    'std_auc': np.std([x['auc'] for x in roccurve_res]),
                    'mean_aar': mean_fpr, 
                    'mean_uar': uar_df.mean(),
                    'pos_std_uar': uar_df.mean() + uar_df.std(),
                    'neg_std_uar': uar_df.mean() - uar_df.std(),
                    'mean_uar_aar_auc': np.mean([x['uar_aar_auc'] for x in roccurve_res]),
                    'std_uar_aar_auc': np.std([x['uar_aar_auc'] for x in roccurve_res])}        
   
        if save_to_file:
            
            f = open(os.path.join(save_dir, 'roc_curve_res_' + u + '.p'), 'w')
            pickle.dump(roccurve_res, f)
            f.close()
            
            f = open(os.path.join(save_dir, 'mean_roc_curve_res_' + u + '.p'), 'w')
            pickle.dump(mean_roccurve_res, f)
            f.close()             
        else:
            all_res[u] = {'folds': roccurve_res, 'mean': mean_roccurve_res}
            
    if not save_to_file:
        return all_res
    


def featobj_model_grid_search_preprocessed(
        estimator, preprocessed_data, params, all_feats, unames, 
        user_fold_key, att_fold_key, train_data_key, 
        valid_data_key, train_samps_key, valid_samps_key, 
        scorer, save_dir, file_add, save_to_file, user_targets = {},
        get_opt_res = True, test_data_key = None, test_samps_key = None):

    """ Uses preprocessed feature data to grid search on model params.
    
    
    Args:
        estimator: A sklearn BaseEstimator object with fit and predict calls
        preprocessed_data: A dictionary of user processed features for use
            with cross validation.  The format is as follows:
                {'User1': [{
                        user_fold_key: An integer indicating the user fold
                        att_fold_key: An integer indicating the attacker fold
                        train_samps_key: A list of sample numbers (from 
                            original feature set containing all users) to 
                            be used in training for this fold,
                        valid_samps_key: A list of sample numbers (from original
                            feature set containing all users) to be used in 
                            testing for this fold,
                        train_data_key: A dataframe or list of dicts of 
                            training features for this fold,
                        valid_data_key: A dataframe or list of dicts of testing 
                            features for this fold},
                            {}, ...],
                
                'User2': [...]}
        params: A dictionary containing lists of param values to iterate
            through and optimize over.
        all_feats: A dictionary containing all raw feature objects for
            all users.  It must contain the following columns:
                'feat_group'
                'feat_name'
                'feat_samp_num'
                'feat_time'
                'feat_type'
                'feat_value'
                'user' 
        unames: A list of the user names to use.
        user_fold_key: A string indicating the user fold key in processed_data.
        att_fold_key: A string indicating the attacker fold key in 
            processed_data.
        train_data_key: A string indicating the training data key in 
            processed_data.
        test_data_key: : A string indicating the testing data key in 
            processed_data.
        train_samps_key: A string indicating the training sample numbers key 
            in processed_data.
        test_samps_key: A string indicating the testing sample numbers key in 
            processed_data.
        scorer: A function in the form fun(ground_truth_labels, 
           predicted_labels, uar_save, aar_save) where:
            
            ground_truth_labels: A list mapping the testing samples to actual 
                classes (-1 for not user, 1 for user)
            predict_labels: A list of predicted classes for the testing samples.
            uar_save: An empty list to save the UAR
            aar_save: An empty list to save the AAR
            
        save_dir: A directory where the results can be saved
        file_add: A string containing info to add to file name
        save_to_file: A boolean indicating if the function returns the 
            results or saves to file.    
        user_targets: An optional dict containing sample number, source_label
            pairs indicating if a sample comes from a user(1) or an attacker(-1),
            for all users                          
    
    Returns:
        If save_to_file is True, it returns nothing and saves a file of
        form '/save_dir/grid_search_file_add_u.p' where file_add is given
        and u is the user, for each user.
        
        If save_to_file is False, this function returns the results.
        
        The results are in the form:
            {'User1':
                {'best_res':{
                        'uar': The averge uar over folds of the best param 
                            combo,
                        'aar': The averge aar over folds of the best param 
                            combo,
                        'params': A dictionary containing the best param combo,
                        'score': The score (from scorer func) of the best 
                            param combo}},
                'all_res': [{
                        'param_combo': An integer uniquely identifying the 
                            param combo (for data splitting)
                        'user_fold': An integer indicating the user fold,
                        'att_fold': An integer indicating the attacker fold,
                        'uar': The uar of this param combo and fold,
                        'score': The score (from scorer func) of this param 
                            combo and fold,
                        'params': The param user in this result,
                        'aar': The aar of this param combo and fold,
                        'samp_scores': The individual scores of the test 
                            samples of the param combo and fold},
                        {}, ....]},
            'User2': {},
            ...}
            
    """
        
    unfit_est = clone(estimator)
    
    # Get param combos
    param_names = params.keys()
    param_combos = list(itertools.product(*[params[k] for k in param_names]))
    param_combos = pd.DataFrame(param_combos, columns = param_names)
    
    print 'num combos: ' + str(param_combos.shape[0])
    
    all_res = {}
    for u in unames:
        print u
        
        # Get preprocessed data info for user
        info = pd.DataFrame([dict((k, x[k]) for k in (user_fold_key, att_fold_key, 
                                  train_data_key, valid_data_key,
                                  train_samps_key, valid_samps_key,
                                  test_samps_key, test_data_key)) \
                                for x in preprocessed_data[u]])
        
        info[user_fold_key] = info[user_fold_key].astype('int')
        info[att_fold_key] = info[att_fold_key].astype('int')
    
        # Get targets
        if len(user_targets.keys()) == 0: 
            targets = featobj_oneclass_targets(all_feats, u, 
                                               'user', 'feat_samp_num')
        else:
            targets = user_targets[u]
    
        grid_search_res = []
        
        for p_num, p_combo in param_combos.iterrows():
            print p_num
            
            est = clone(unfit_est)
            res = est.set_params(**dict(p_combo))            

            for user_fold, g_user_fold in info.groupby(user_fold_key):
                #print user_fold
                
                # Train model
                train_samps = g_user_fold[train_samps_key].iloc[0]
                train_data = parse_features_data(
                        g_user_fold[train_data_key].iloc[0])
                est.fit(train_data)
                
                # Run training data through to get training scores:
                train_pred = est.predict(train_data)
                train_scores = zip(train_samps, train_pred)  
                train_extra_info = est.get_pred_extra_info()
                
                
                for att_fold, g_att_fold in g_user_fold.groupby(att_fold_key):
                    #print att_fold
                                        
                    # Predict on model - test data
                    test_samps = g_att_fold[valid_samps_key].iloc[0]
                    
                    test_data = parse_features_data(
                        g_att_fold[valid_data_key].iloc[0],
                        test_samps)

                    val_pred = est.predict(pd.DataFrame(test_data))
                    samp_scores = zip(test_samps, val_pred)
                    extra_info = est.get_pred_extra_info()
                    
                    uar = []
                    aar = []
                    param_score = scorer(targets, samp_scores,
                                                  uar, aar)
                    
                    
                    # If testing key is present, predict on test data
                    test_scores = []
                    test_extra_info = {}
                    if test_data_key != None:
                        test_samps = g_att_fold[test_samps_key].iloc[0]
                        
                        test_data = parse_features_data(
                            g_att_fold[test_data_key].iloc[0],
                            test_samps)                        
                        
                        test_pred = est.predict(pd.DataFrame(test_data))
                        test_scores = zip(test_samps, test_pred)
                        test_extra_info = est.get_pred_extra_info()
                    


                    grid_search_res.append({'user_fold': user_fold,
                                            'att_fold': att_fold,
                                            'params': dict(p_combo),
                                            'param_combo': p_num,
                                            'samp_scores': samp_scores,
                                            'test_scores': test_scores,
                                            'train_scores': train_scores,
                                            'uar': uar[0],
                                            'aar': aar[0],
                                            'score': param_score,
                                            'extra_info': extra_info,
                                            'train_extra_info': train_extra_info,
                                            'test_extra_info': test_extra_info})

    
        # Get best param set
        if get_opt_res:
            grid_search_res_df = pd.DataFrame(grid_search_res)
            
            res = []
            att_fold_grpd = grid_search_res_df.groupby('param_combo')
            for ind,x in att_fold_grpd:
                res.append(x.groupby('user_fold').aggregate(np.mean))
            
            res = pd.concat(res)
            
            best_res = res.groupby('param_combo').aggregate(np.mean)
            
            best_combo_num = best_res['score'].idxmax()
            final_best_res = dict(best_res.loc[best_combo_num,:])
            final_best_res['params'] = grid_search_res_df.loc[
                    grid_search_res_df['param_combo'] == \
                    best_combo_num, 'params'].iloc[0]
        else:
            final_best_res = {}
        
        final_res = {'best_res': final_best_res,
                    'all_res': grid_search_res}
        
        if save_to_file:            
            f = open(os.path.join(
                    save_dir, 'grid_search_' + file_add + '_' + u + '.p'), 'w')
            pickle.dump(final_res, f)
            f.close()         
        else:
            all_res[u] = final_res
            
    if not save_to_file:
        return all_res        
    
def get_optimal_res_grid_search(grid_search_res_df):
    res = []
    att_fold_grpd = grid_search_res_df.groupby('param_combo')
    for ind,x in att_fold_grpd:
        res.append(x.groupby('user_fold').aggregate(np.mean))
    
    res = pd.concat(res)
    
    best_res = res.groupby('param_combo').aggregate(np.mean)
    
    best_combo_num = best_res['score'].idxmax()
    final_best_res = dict(best_res.loc[best_combo_num,:])
    
    ind = grid_search_res_df['param_combo'] == best_combo_num
    
    final_best_res['params'] = grid_search_res_df.loc[ind, 'params'].iloc[0]   
    final_best_res['best_combo'] = best_combo_num
    final_best_res['extra_info'] = grid_search_res_df.loc[ind, 'extra_info'].iloc[0]
    final_best_res['test_extra_info'] = grid_search_res_df.loc[ind, 'test_extra_info'].iloc[0]  
    final_best_res['train_extra_info'] = grid_search_res_df.loc[ind, 'train_extra_info'].iloc[0]        
    return final_best_res

def process_grid_search_res_single_samp_pred(grids_res_files, unames, 
                                             user_targets, opt_grids_res,
                                             other_decision_key = None,
                                             strat_sample_seed = None):

    ''' Processes the results from gridsearch, where samples were
    predicted one at a time.
    
    This function collapses the results into a more compact data frame
    of results.  It also calculates the optimal grid search param combo.
    Optionally, the uar/aar for each combo can be re-calculated using
    another intermediate decision made by the engine (in 'extra info')
    '''
    
    grids_res = {}
    for u in unames:
        print u
        f = open(grids_res_files[u], 'r')
        gs_res = pickle.load(f)
        f.close()
    
        res = pd.DataFrame(gs_res['all_res'])
        
        # Sample user and attacker samples such that they are equal in size
        all_samp_nums = []
        if strat_sample_seed != None:
            random.seed(strat_sample_seed)
            
            targ_dict = dict(user_targets[u])
            samp_nums = np.unique(np.array([x[0][0] for x in res['samp_scores']]))
            samp_targs = np.array([targ_dict[x] for x in samp_nums])
            user_samps = samp_nums[samp_targs == 1]
            att_samps = samp_nums[samp_targs == -1]
            
            if len(user_samps) > len(att_samps):
                user_samps = random.sample(user_samps, len(att_samps))
            elif len(user_samps) < len(att_samps):
                att_samps = random.sample(att_samps, len(user_samps))        
                
            all_samp_nums = list(user_samps) + list(att_samps)
            print len(all_samp_nums)
        
        grids_res[u] = []
        for ind,g in res.groupby('param_combo'):
            new_g = dict(g.iloc[0,:].copy())
            
            samp_inds = []
            if strat_sample_seed == None:
                samp_scores = [x[0] for x in g['samp_scores']]
                samp_inds = range(len(g['samp_scores']))
            else:
                samp_scores = [x[0] for x in g['samp_scores'] if x[0][0] in all_samp_nums]
                samp_inds = [i for i in range(len(g['samp_scores'])) if g['samp_scores'].iloc[i][0][0] in all_samp_nums]
    
            new_g['samp_scores'] = samp_scores
            uar = []
            aar = []
            sc = ave_uar_aar_featobj(user_targets[u], samp_scores, uar, aar)
            new_g['uar'] = uar[0]
            new_g['aar'] = aar[0]
            new_g['score'] = sc
            
            extra_info = []
            for x in list(g['extra_info']):
                y = {}
                for k in x.keys():
                    if ((isinstance(x[k], list)) or\
                        (isinstance(x[k], np.ndarray))) and\
                        (len(x[k]) == 1):
                        y[k] = x[k][0]
                    else:
                        y[k] = x[k]
                extra_info.append(y)
                
            # Sample extra info according to test sampling
            new_extra_info = []
            for i in samp_inds:
                new_extra_info.append(extra_info[i])
                
            if other_decision_key != None:
                pred = np.array([x[other_decision_key][0] for x in g['extra_info']])
                                 
                tar = dict(user_targets[u])
                samp_nums = [x[0] for x in samp_scores]
                
                conds = np.array([tar[x] for x in samp_nums])
                new_g['uar'] = float(len(pred[(pred == 1) & (conds == 1)])) / float(len(pred[conds == 1]))
                new_g['aar'] = float(len(pred[(pred == 1) & (conds == -1)])) / float(len(pred[conds == -1]))
                new_g['score'] = np.mean([new_g['uar'], 1.0 - new_g['aar']])         
                         
            
            new_g['extra_info'] = pd.DataFrame(new_extra_info).to_dict('list')
            grids_res[u].append(new_g)
        opt = get_optimal_res_grid_search(pd.DataFrame(grids_res[u]))
        opt['user'] = u
        opt_grids_res.append(opt)
    return grids_res

def get_optimal_scores_gt_grid_search_res(opt_grids_res, unames, targets, 
                                          test_samp_nums, raw_score_key,
                                          mult_100 = True):

    ''' Extracts raw scores and ground truth labels from optimal grid
    search results.
    
    '''
    
    res = {}
    for u in unames:
        print u
        tar = dict(targets[u])
        scores = opt_grids_res['extra_info'].loc[opt_grids_res['user'] == u].iloc[0][raw_score_key]
        
        if mult_100:
            scores = [s * 100.0 for s in scores]
        
        conds = []
        for x in test_samp_nums[u]:
            conds.append(tar[x])
        
        res[u] = {'scores': scores, 'conds': conds}
        
    return res

def get_optimal_scores_gt_grid_search_res_train_valid_test(
        opt_grids_res, unames, targets, test_samp_nums, raw_score_key,
        mult_100 = True):

    ''' Extracts raw scores and ground truth labels from optimal grid
    search results.  It also extracts the training and testing scores.
    
    '''
    
    if mult_100:
        mult = 100.0
    else:
        mult = 1.0
    
    res = {}
    for u in unames:
        print u
        
        # Get main validation results
        
        tar = dict(targets[u])
        
        scores = [x * mult for x in\
                  opt_grids_res['extra_info'].loc[opt_grids_res['user'] == u].\
                  iloc[0][raw_score_key]]
        
        conds = [tar[x] for x in test_samp_nums[u]['validate']]
            
        test_scores = [x * mult for x in\
                  opt_grids_res['test_extra_info'].loc[opt_grids_res['user'] == u].\
                  iloc[0][raw_score_key]]
        
        test_conds = [tar[x] for x in test_samp_nums[u]['test']]        
        
        train_scores = [x * mult for x in\
                  opt_grids_res['train_extra_info'].loc[opt_grids_res['user'] == u].\
                  iloc[0][raw_score_key]]
     
        
        res[u] = {'scores': scores, 'conds': conds,
           'test_scores': test_scores, 'test_conds': test_conds,
           'train_scores': train_scores}
        
    return res

def featobj_model_runthrough_preprocessed(
        estimator, preprocessed_data, user_params, all_feats, unames, 
        user_fold_key, att_fold_key, train_data_key, 
        test_data_key, train_samps_key, test_samps_key, 
        scorer, save_dir, file_add, save_to_file, user_targets = {}):

    """ Uses preprocessed feature data to run through the model
    
    
    Args:
        estimator: A sklearn BaseEstimator object with fit and predict calls
        preprocessed_data: A dictionary of user processed features for use
            with cross validation.  The format is as follows:
                {'User1': [{
                        user_fold_key: An integer indicating the user fold
                        att_fold_key: An integer indicating the attacker fold
                        train_samps_key: A list of sample numbers (from 
                            original feature set containing all users) to 
                            be used in training for this fold,
                        test_samps_key: A list of sample numbers (from original
                            feature set containing all users) to be used in 
                            testing for this fold,
                        train_data_key: A dataframe or list of dicts of 
                            training features for this fold,
                        test_data_key: A dataframe or list of dicts of testing 
                            features for this fold},
                            {}, ...],
                
                'User2': [...]}
        all_feats: A dictionary containing all raw feature objects for
            all users.  It must contain the following columns:
                'feat_group'
                'feat_name'
                'feat_samp_num'
                'feat_time'
                'feat_type'
                'feat_value'
                'user' 
        unames: A list of the user names to use.
        user_fold_key: A string indicating the user fold key in processed_data.
        att_fold_key: A string indicating the attacker fold key in 
            processed_data.
        train_data_key: A string indicating the training data key in 
            processed_data.
        test_data_key: : A string indicating the testing data key in 
            processed_data.
        train_samps_key: A string indicating the training sample numbers key 
            in processed_data.
        test_samps_key: A string indicating the testing sample numbers key in 
            processed_data.
        scorer: A function in the form fun(ground_truth_labels, 
           predicted_labels, uar_save, aar_save) where:
            
            ground_truth_labels: A list mapping the testing samples to actual 
                classes (-1 for not user, 1 for user)
            predict_labels: A list of predicted classes for the testing samples.
            uar_save: An empty list to save the UAR
            aar_save: An empty list to save the AAR
            
        save_dir: A directory where the results can be saved
        file_add: A string containing info to add to file name
        save_to_file: A boolean indicating if the function returns the 
            results or saves to file.    
        user_targets: An optional dict containing sample number, source_label
            pairs indicating if a sample comes from a user(1) or an attacker(-1),
            for all users                          
    
    Returns:
        If save_to_file is True, it returns nothing and saves a file of
        form '/save_dir/grid_search_file_add_u.p' where file_add is given
        and u is the user, for each user.
        
        If save_to_file is False, this function returns the results.
        
        The results are in the form:
            {'User1':
                {'best_res':{
                        'uar': The averge uar over folds of the best param 
                            combo,
                        'aar': The averge aar over folds of the best param 
                            combo,
                        'params': A dictionary containing the best param combo,
                        'score': The score (from scorer func) of the best 
                            param combo}},
                'all_res': [{
                        'param_combo': An integer uniquely identifying the 
                            param combo (for data splitting)
                        'user_fold': An integer indicating the user fold,
                        'att_fold': An integer indicating the attacker fold,
                        'uar': The uar of this param combo and fold,
                        'score': The score (from scorer func) of this param 
                            combo and fold,
                        'params': The param user in this result,
                        'aar': The aar of this param combo and fold,
                        'samp_scores': The individual scores of the test 
                            samples of the param combo and fold},
                        {}, ....]},
            'User2': {},
            ...}
            
    """
        
    unfit_est = clone(estimator)
    
    all_res = {}
    for u in unames:
        print u
        
        # Get preprocessed data info for user
        info = pd.DataFrame([dict((k, x[k]) for k in (user_fold_key, att_fold_key, 
                                  train_data_key, test_data_key,
                                  train_samps_key, test_samps_key)) \
                                for x in preprocessed_data[u]])
        
        info[user_fold_key] = info[user_fold_key].astype('int')
        info[att_fold_key] = info[att_fold_key].astype('int')
    
        # Get targets
        if len(user_targets.keys()) == 0: 
            targets = featobj_oneclass_targets(all_feats, u, 
                                               'user', 'feat_samp_num')
        else:
            targets = user_targets[u]
    

        est = clone(unfit_est)
        est.set_params(**user_params[u])          
        user_res = []       
        for user_fold, g_user_fold in info.groupby(user_fold_key):
            #print user_fold
            
            # Train model
            train_data = parse_features_data(
                    g_user_fold[train_data_key].iloc[0])
            est.fit(train_data)
            
            for att_fold, g_att_fold in g_user_fold.groupby(att_fold_key):
                #print att_fold
                                    
                # Predict on model - test data
                test_samps = g_att_fold[test_samps_key].iloc[0]
                
                test_data = parse_features_data(
                    g_att_fold[test_data_key].iloc[0],
                    test_samps)
                val_pred = est.predict(pd.DataFrame(test_data))

                samp_scores = zip(test_samps, val_pred)
                
                uar = []
                aar = []
                param_score = scorer(targets, samp_scores,
                                              uar, aar)

                user_res.append({'user_fold': user_fold,
                                        'att_fold': att_fold,
                                        'samp_scores': samp_scores,
                                        'params': user_params[u],
                                        'uar': uar[0],
                                        'aar': aar[0],
                                        'score': param_score})


        
        if save_to_file:            
            f = open(os.path.join(
                    save_dir, 'model_runthrough' + file_add + '_' + u + '.p'), 'w')
            pickle.dump(user_res, f)
            f.close()         
        else:
            all_res[u] = user_res
            
    if not save_to_file:
        return all_res   
    
def featobj_oneclass_targets(all_feats, u, user_col, samp_num_col):
    """Returns the one class targets with respect to a user.
    
    This function returns a 1 for each sample in all_feats belonging to
    to the user and -1 otherwise.
    
    This function accepts a dataframe of feat objects (as opposed to
    the traditional feature dataframe with the feat names on the columns)

    Args:
        all_feats: A dataframe of feature objects.  It must have a user
            user column and a sample number column
        u: The name of the user with respect to which the targets are generated
        user_col: The name of the user column
        samp_num_col: The name of the sample column
    Returns:
        A list of tuples of the form (sample number, target)
    """      
    
    targets = []
    found_samp_nums = []
    for __,row in all_feats.iterrows():
        if row[user_col] == u:
            tar = 1
        else:
            tar = -1
        if row[samp_num_col] not in found_samp_nums:
            targets.append((row[samp_num_col], tar))
            found_samp_nums.append(row[samp_num_col])
    return targets

def parse_features_data(raw_feat_data, dict_order = None):
    """Parses a set of features.
    
    This function determines if the data is stored in a dictionary,
    in an array as a dataframe, a list of dictionaries or a list of 
    dataframes, and processes it appropriately to produce a single data frame

    Args:
        raw_feat_data: the raw feature data.
        samp_nums: A list of keys to indicate the order of processing
            if the data is a dictionary.
    Returns:
        A data frame of feature data (columns are the features, rows
        are the samples)
    """          
    data = copy.deepcopy(raw_feat_data)
    if isinstance(data, list):
        if len(data) == 1:
            data = data[0]
        else:
            if isinstance(data[0], pd.DataFrame):
                data = pd.concat(data)
                data.index = range(data.shape[0])
            else:
                data = pd.DataFrame(data, index = range(len(data)))
                
    elif isinstance(data, dict):
        
        # Extract individual samples into a list
        if isinstance(data[dict_order[0]], list):
            data = [data[k][0] for k in dict_order]
        else:
            data = [data[k] for k in dict_order]
        
        # Determine if the each sample is a dataframe or dict
        # and process accordingly
        if isinstance(data[0], pd.DataFrame):
            data = pd.concat(data)
            data.index = range(data.shape[0])
        else:
            data = pd.DataFrame(data, index = range(len(data)))       
    return data
    
            
def ave_uar_aar_featobj(ground_truth, predictions, uar_save = [], aar_save = []):
    """A custom scorer to be used with results from a feat obj model.
    
    This scorer takes lists of tuples for ground truth and predictions,
    containing the sample number and the score.  This allows scoring
    of models handling feature objects as input.
    
    This scorer outputs the mean of the UAR and 1 - AAR.

    Args:
        ground_truth: A list of tuples containing the sample number
            and either 1 (for user sample) or -1. (for attacker sample) This
            does not need to be the same size as predictions - It just needs
            to contain all the sample numbers in predictions.
        predictions: A list of tuples containing the sample number
            and either 1 (for user sample) or -1 (for attacker sample)            
    Returns:
        The mean of the UAR and 1 - AAR for the predictions.
    """  
       
    samp_nums_pred = np.array([x[0] for x in predictions])
    samp_nums_gt = np.array([x[0] for x in ground_truth])

    gt_inds = um.match_arrays(np.unique(samp_nums_pred), samp_nums_gt)
    gt = np.array([ground_truth[i][1] for i in gt_inds])
    
    pt_inds = um.match_arrays(np.unique(samp_nums_pred), samp_nums_pred)
    pred = np.array([predictions[i][1] for i in pt_inds])
    
    num_user = len(pred[(gt > 0.0)])
    num_att = len(pred[(gt < 0.0)])
    
    if num_user == 0:
        uar = np.nan
    else:
        uar = float(len(pred[(gt > 0.0) & (pred > 0.0)])) / float(num_user)
        
        
    if num_att == 0:
        aar = np.nan
    else:
        aar = float(len(pred[(gt < 0.0) & (pred > 0.0)])) / float(num_att)        
    
    
    #print 'pred samps: '
    #print np.unique(samp_nums_pred)
    #print 'uar: ' + str(uar), 'aar: ' + str(aar)
    
    uar_save.append(uar)
    aar_save.append(aar)
        
    
    return np.mean([uar, 1.0 - aar])

def uar_aar_score_cond(scores, thresh, gt, user_lab, att_lab, 
                       bigger_score_better = True):
    """Determines uar aar using scores, ground truth and a threshold

    Args:
        scores: A list of scores (0 to 100)
        thresh: A threshold to use on the scores for authentication
        gt: A list of labels indicating user or not user
        user_lab: The label in gt indicating the user
        att_lab: The label in gt indiciating not the user (attacker)          
    Returns:
        A tuple containing the uar and aar
    """  
    
    scores_arr = np.array(scores)
    gt_arr = np.array(gt)
    
    user_bool = gt_arr == user_lab
    att_bool = gt_arr == att_lab
    
    if bigger_score_better:
        auth = scores_arr >= thresh
    else:
        auth = scores_arr <= thresh        
    uar = float(len(gt_arr[user_bool & auth])) / float(len(gt_arr[user_bool]))
    aar = float(len(gt_arr[att_bool & auth])) / float(len(gt_arr[att_bool]))
    
    return uar, aar

def plot_roc_curves_res(res, title, add_std = False):
    """Creates a plotly object of ROC curves
    
    This function takes the results of 'featobj_model_roc_curve_preprocessed'
    for one user and plots ROC curves for each fold, the mean and the 
    mean +/- std. The legend contains the AUC values
    

    Args:
        res: A dictionary containing the output of 
            'featobj_model_roc_curve_preprocessed' for one user
        title: The title of the plot
        add_std: A boolean, if true the mean +/- sd curve is added.            
    Returns:
        A plotly figure object
    """  
    
    traces = []

    for i in range(len(res['folds'])):
        traces.append(go.Scatter(x = res['folds'][i]['aar'], 
                                             y = res['folds'][i]['uar'],
                                            name = 'Fold ' + str(i) + ' ROC, AUC = ' +\
                                             '%.3f' % res['folds'][i]['uar_aar_auc'],
                                            line = dict(color = '#0000FF', width = 1, dash = 'dash'),
                                            mode = 'lines'))    

    if add_std:
        traces.append(go.Scatter(x = res['mean']['mean_aar'], 
                                             y = res['mean']['pos_std_uar'],
                                            line = dict(color = 'grey'),
                                            showlegend = False))
    
        traces.append(go.Scatter(x = res['mean']['mean_aar'], 
                                             y = res['mean']['neg_std_uar'],
                                            name = '+/- 1 std', fill='tonexty',
                                            line = dict(color = 'grey')))

    traces.append(go.Scatter(x = res['mean']['mean_aar'], 
                                         y = res['mean']['mean_uar'],
                                        name = 'Mean ROC, AUC = ' +\
                                         '%.3f' % res['mean']['mean_uar_aar_auc'] +\
                                         ' +/- ' + '%.3f' % res['mean']['std_uar_aar_auc'],
                                        line = dict(color = 'black')))

    return go.Figure(data = traces, layout = go.Layout(title = title))           
                        
            
def plot_learning_curves_res(res, title, uar_key = 'val_uar', aar_key = 'val_aar'):
    """Creates a plotly object of learning curves
    
    This function takes the results of 'featobj_model_learning_curve_preprocessed'
    for one user and plots the mean learning curves.
    

    Args:
        res: A dictionary containing the output of 
            'featobj_model_learning_curve_preprocessed' for one user
        title: The title of the plot          
    Returns:
        A plotly figure object
    """  
    
    res_df = pd.DataFrame(res)
    
    # Get mean values over folds
    mean_res = res_df.groupby('train_size').aggregate(np.mean)
    
    traces = [go.Scatter(x = mean_res.index, y = mean_res['train_uar'], 
                  name = 'training score', mode = 'lines', line = dict(color = 'blue')),
      go.Scatter(x = mean_res.index, y = mean_res[uar_key], 
                  name = 'validation uar', mode = 'lines', line = dict(color = 'red')),
      go.Scatter(x = mean_res.index, y = mean_res[aar_key], 
                  name = 'validation aar', mode = 'lines', line = dict(color = 'green'))]
    return go.Figure(data = traces, layout = go.Layout(title = title,
                                                       yaxis = dict(range = [0,1])))    

def plot_learning_curve_folds(res, title, showleg = True, 
                              uar_key = 'val_uar', aar_key = 'val_aar'):
    """Creates a plotly object of learning curves for each fold.
    
    This function takes the results of 'featobj_model_learning_curve_preprocessed'
    for one user and plots all fold learning curves.
    

    Args:
        res: A dictionary containing the output of 
            'featobj_model_learning_curve_preprocessed' for one user
        title: The title of the plot          
    Returns:
        A plotly figure object
    """  
    
    res_df = pd.DataFrame(res)
    
    # Go through each fold and plot the training, validation uar and validation
    # aar
    traces = []
    num_plts = 0
    for inds, g in res_df.groupby(['user_fold', 'att_fold']):
        
        if num_plts == 0:
            show_leg = True & showleg
        else:
            show_leg = False & showleg
        
        num_plts = num_plts + 1
        traces.extend([go.Scatter(x = g['train_size'], y = g['train_uar'], 
                      name = 'training score', mode = 'lines', line = dict(color = 'black'),
                      showlegend = show_leg),
          go.Scatter(x = g['train_size'], y = g[uar_key], 
                      name = 'validation uar', mode = 'lines', line = dict(color = 'blue'),
                      showlegend = show_leg),
          go.Scatter(x = g['train_size'], y = g[aar_key], 
                      name = 'validation aar', mode = 'lines', line = dict(color = 'red'),
                      showlegend = show_leg)])  

        num_plts = num_plts + 1
    return go.Figure(data = traces, layout = go.Layout(title = title,
                                                       yaxis = dict(range = [0,1])))          


def thresh_at_uar(scores, conds, user_lab, att_lab, uar):
    arr_scores = np.array(scores)
    arr_conds = np.array(conds)

    user_scores = arr_scores[arr_conds == user_lab]
    
    # Get num samples corresponding to uar
    num_samp = int(uar * len(user_scores))
    if num_samp == len(user_scores):
        return min(user_scores)
    
    sort_user_sc = sorted(user_scores, reverse = True)
    return sort_user_sc[num_samp - 1]

def get_roc_curve_info(scores, labels, user_lab, att_lab, thresholds,
                      bigger_score_better = True, opt_thresh_func = None):

    roc_res = {}
    
    # Set labels to correct range
    arr_labels = np.array(labels)
    gr_truth = np.array(labels)
    gr_truth[arr_labels == user_lab] = 1
    gr_truth[arr_labels == att_lab] = -1    
    
    arr_scores = np.array(scores)
    
    # Get list of thresholds
    if len(thresholds) == 0:
        fpr, tpr, new_thresholds = roc_curve(gr_truth, scores)

        for t in new_thresholds:
            thresholds.append(t)

    # Get uar and aar
    uar = []
    aar = []
    for t in thresholds:
        uar_new, aar_new = uar_aar_score_cond(arr_scores, t, arr_labels, 
                                                  user_lab, att_lab,
                                                  bigger_score_better)
        uar.append(uar_new)
        aar.append(aar_new)
        
    auc_val = auc(aar, uar)
        

    roc_res['thresh'] = thresholds
    roc_res['uar'] = uar
    roc_res['aar'] = aar
    roc_res['auc'] = auc_val
    
    # Get optimal UAR/AAR
    if opt_thresh_func == None:
        opt_res = get_optimal_roc_res_mean(uar, aar, arr_scores, arr_labels, 
                                           thresholds)
    else:
        opt_res = opt_thresh_func(uar, aar, arr_scores, arr_labels, 
                                           thresholds)        

    roc_res['opt_res'] = opt_res
    
    
    return roc_res

def get_optimal_roc_res_mean(uar, aar, scores, gt, thresholds):
    
    """Calculates optimal roc curve results based on the mean of UAR/AAR
    

    Args:
        uar: A list of uar values
        aar: A list of aar values
        thresholds: A list of thresholds        
    Returns:
        A dictionary containing optimal results
    """  
    
    
    m_uar_aar = np.array([np.mean([x[0], 1.0 - x[1]]) for x in zip(uar, aar)])
    
    max_ind = np.argmax(m_uar_aar)
    
    opt_res = {'optimal_thresh': np.array(thresholds)[max_ind],
                     'optimal_uar': np.array(uar)[max_ind],
                     'optimal_aar': np.array(aar)[max_ind]}

    return opt_res

def get_optimal_roc_res_uar(uar, aar, scores, gt, thresholds, target_uar):
    
    """Calculates optimal roc curve results based a target uar
    

    Args:
        uar: A list of uar values
        aar: A list of aar values
        thresholds: A list of thresholds        
    Returns:
        A dictionary containing optimal results
    """  
    
    scores_arr = np.array(scores)
    gt_arr = np.array(gt)
    
    user_scores = scores_arr[gt_arr == 1]
    att_scores = scores_arr[gt_arr == -1]
    
    sort_user_sc = sorted(user_scores, reverse = True)
    
    # Get closest to target_uar without going over
    samp_num = int(float(len(user_scores)) * target_uar)
    
    thresh = sort_user_sc[samp_num - 1]
    
    opt_res = {'optimal_thresh': thresh,
               'optimal_uar': float(samp_num) / float(len(user_scores)),
               'optimal_aar': float(len(att_scores[att_scores >= thresh])) / float(len(att_scores))}

    return opt_res  

def get_feats_from_processed_lc_data(data_dir, f_names, unames, tr_size, data_keys, 
                                     samp_keys, cond_vals, user_targets,
                                     repl_names):
    
    user_feats = {}
    for u in unames:
        f = open(os.path.join(data_dir, f_names[u]), 'r')
        data = pickle.load(f)
        f.close() 
        
        tr_sizes = np.array([x['train_size'] for x in data])
        ind = np.where(tr_sizes == tr_size)[0][0]
        
        data_pt = data[ind]
        
        targs = dict(user_targets[u])
        
        feats = []
        for i in range(len(data_keys)):
            df = pd.DataFrame(data_pt[data_keys[i]])
            
            if cond_vals[i] == None:
                conds = [targs[x] for x in data_pt[samp_keys[i]]]    
            else:
                conds = [cond_vals[i]] * df.shape[0]
            df['cond'] = conds
            
            feats.append(df)
            
        feats = pd.concat(feats)
        if 'feat_samp_num' in feats.columns:
            del feats['feat_samp_num']
        
        feats.columns = um.change_feat_names(feats.columns, repl_names)
        user_feats[u] = feats
    return user_feats
        
