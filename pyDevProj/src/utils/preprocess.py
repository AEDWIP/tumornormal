'''
Created on Feb 16, 2018

@author: andrewdavidson
'''
import numpy as np

def normalizeData(X):
    """scales data such that mean=0 and standard deviation=1"""
    n,m = X.shape
    
    # axis=1 cause calculation to run across rows
    means = np.mean(X, axis=1).reshape(n, 1)
    std = np.std(X, axis=1).reshape(n, 1)
    #mean, std = tf.nn.moments(X, axes=[1])
    
    centered = X - means
    return (centered / std)
