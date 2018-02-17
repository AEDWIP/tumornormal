'''
Created on Feb 6, 2018

@author: andrewdavidson
'''

from keras.utils.io_utils import HDF5Matrix
import numpy as np

def check_data(X, y, DEBUG=False):
    " raises assertion if  X, and y, dimensions are consistent"
    
    n,m = X.shape
    assert y.shape == (1, m), "y shape failed"
  
    expectedNumFeatures = 2 if DEBUG else 60498
    msg = "expected:" + str(expectedNumFeatures) + " actual:" + str(n)
    assert (n == expectedNumFeatures), msg

def full_train_data_set(input_file):
    print("Training on full dataset")
    X_train = HDF5Matrix(input_file, "X_train")
    y_train = HDF5Matrix(input_file, "y_train")
    return [X_train, y_train]

def full_test_data_set(input_file):
    print("Testing on full dataset")
    X_test = HDF5Matrix(input_file, "X_test")
    y_test = HDF5Matrix(input_file, "y_test")
    return [X_test, y_test]

def small_train_data_set(input_file):
    print("Training on partial dataset")
    X_train = HDF5Matrix(input_file, "X_train", start=0, end=1000)
    y_train = HDF5Matrix(input_file, "y_train", start=0, end=1000)
    return [X_train, y_train]

def small_test_data_set(input_file):
    print("Testing on partial dataset")
    X_test = HDF5Matrix(input_file, "X_test", start=0, end=200)
    y_test = HDF5Matrix(input_file, "y_test", start=0, end=200)
    return [X_test, y_test]

def test_train_data_set():
    # this data set is simple enough we can do the math on paper
    print("Training on trival (debug) data set")
    X_train = np.array([
                        [ 1.,  2.],
                        [ 3.,  4.],
                        [ 5.,  6.],
                        [ 7.,  8.],
                        [ 9., 10.],
                        [11., 12.],
                        [13., 14]
                            ])
    
    # y_train must be a rank 2 array
    # i.e. we want transpose of y_train to be a column vector not
    # an array
    y_train = np.array( [[0., 0., 0., 1., 1., 1., 1.]] )
    
    return [X_train, y_train]
  
def test_test_data_set():
    # this data set is simple enough we can do the math on paper
    X_train, y_train = test_train_data_set()
    X_test = X_train * 10
    y_test = y_train
    
    return [X_test, y_test]
    
