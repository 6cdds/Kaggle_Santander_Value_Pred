# -*- coding: utf-8 -*-
"""
Created on Fri May 26 16:53:33 2017

This script contains utility methods - i.e. general methods that are used
to facilitate data analysis

@author: colleen.s
"""

import numpy as np
import pandas
from math import floor, sqrt, atan2
from scipy.signal import butter, lfilter

def match_arrays(array1, array2):
    """Determines the indices of the elements of array1 within array2

    Args:
        array1: A numpy array
        array2: A numpy array
    Returns:
        A numpy array of indices with length equal to len(array1)

    """

    vfunc = np.vectorize(lambda x: find_in_array(x, array2))
    return vfunc(array1)
    

def find_in_array(val, array):
    """Determines the first index of val within array

    Args:
        val: A value to search for
        array: A numpy array to search in
    Returns:
        An int if val is found, NaN otherwise

    """
    indices = np.where(array == val)[0]
    if len(indices) == 0:
        return np.nan
    elif len(indices) > 1:
        return int(indices[0])
    else:        
        return int(indices)

def get_window_inds(len_data, len_window, perc_overlap):
    """Gets the indices of a vector after windowing with overlap

    Args:
        len_data: The length of the vector (int)
        len_window: The length of the window (int)
        perc_overlap: A float between 0 and 1, indicating the % overlap of
        windows
    Returns:
        A numpy ndarray containing the indices for each window

    """
    len_overlap = int(round(perc_overlap * len_window))
    len_offset = len_window - len_overlap
    num_windows = int(floor((len_data - len_window) / len_offset)) + 1
    wind_start_inds = [(len_offset * (x - 1)) for x in range(1, num_windows + 1)]
    window_inds = list()
    for x in wind_start_inds:
        window_inds.append(range(x, x + len_window))
    return np.array(window_inds)
    
    
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y        
    
    
def cart2sph(x,y,z):
    rho = sqrt(x**2 + y**2 + z**2)            
    theta = np.arccos(z / rho)    
    phi = np.arctan2(y, x)                          
    return rho, theta, phi

def cart2sph_df(df):
    return pandas.DataFrame([cart2sph(row['x'],row['y'],row['z']) 
                             for __, row in df.iterrows()],
    columns=['rho', 'theta', 'phi'], index=df.index)
    
def get_successive_diff(vals):
    """Get list of difference between successive values in a list.
    
    Args:
        vals: A array-like container of values       

    Returns:
        A list of successive differences of vals

    """
    
    ar_vals = np.array(vals)   
    return [y-x for x, y in zip(ar_vals[:-1], ar_vals[1:])]

def get_pearson_corr(x, y):
    """Get pearson correlation between two lists.
    
    This implementation matches that in the swipe C++ library.
    
    Args:
        x: the first list.
        y: the second list.

    Returns:
        A correlation value.

    """    
    min_len = min([len(x), len(y)])

    newx = np.array(x)[range(min_len)]
    newy = np.array(y)[range(min_len)]
  
    sumx = sum(newx)
    sumy = sum(newy)
    sumxy = sum([i * j for i,j in zip(x,y)])
    sumxx = sum([i**2 for i in x])
    sumyy = sum([i**2 for i in y])
  
    s1 = (float(min_len)*sumxy) - (sumx*sumy)
    s2 = (float(min_len)*sumxx) - (sumx*sumx)
    s3 = (float(min_len)*sumyy) - (sumy*sumy)
  
    if s2 >= 0 and s3 >= 0:
        s4 = np.sqrt(s2 * s3)
    else:
        s4 = 0
  
    corr_val = 0
    if s4 != 0:
        corr_val = s1 / s4
    
    return corr_val

def get_pearson_corr_slope(vals):
    """Get the slope of a line using pearson correlation
    
    This implementation matches that in the swipe C++ library. Only the 
    y-values are given since the x-values are assumed to have dx = 1.
    
    Args:
        vals: The y-values

    Returns:
        A slope value.

    """        
    inds = range(len(vals))
    corr = get_pearson_corr(inds, vals)
    slope = corr * np.std(vals) / np.std(inds)
    return slope

  
def get_condition_ids(cond_trans_ids, trans_ids):
    """Created condition ids given transaction ids and conditions
    
    This function is used to split data into different data collection
    conditions based on a transaction id.
    
    Args:
        cond_trans_ids: A dictionary indicating the start and end
        transaction ids for certain conditions. Eg.
            {'User1': [
                        {'condition_id': 1, 
                          'start_id': '5ae72bcc76496c4d883f36c3',
                          'end_id': '5aeb4d3576496c4d883f3bcd'},
                        {'condition_id': 2, 
                          'start_id': '5aec9fcf76496c4d883f3cb5',
                          'end_id': '5aeca6d376496c4d883f3e3b'}]}

    Returns:
        A list of condition ids

    """      
    
    cond_ids = [np.nan] * len(trans_ids)
    
    trans_ids_arr = np.array(trans_ids)

    for cond in cond_trans_ids:
        start_ind = np.where(trans_ids_arr == cond['start_id'])[0][0]
        end_ind = np.where(trans_ids_arr == cond['end_id'])[0][0]
        
        cond_ids[start_ind : (end_ind + 1)] = \
            [cond['condition_id']] * (end_ind + 1 - start_ind)
            
    return cond_ids 

def change_feat_names(feat_names, repl_names):
    ''' Replaces parts of feature names with different names

    This function can be used to shorten feature names or create more
    descriptive names

    '''    
    
    new_feat_names = []
    for f in feat_names:
        new_f = f
        for b in repl_names.keys():
            if b in f:
                new_f = f.replace(b, repl_names[b])
        new_feat_names.append(new_f)
    return new_feat_names   