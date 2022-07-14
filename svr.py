from re import M
from sklearn.svm import SVR, LinearSVR, NuSVR
from sklearn.multioutput import MultiOutputRegressor
import numpy as np
from sklearn.metrics import make_scorer
from scipy.stats import uniform, randint
import pandas as pd
import pickle

def calc_scores (x, y):
    # Taken from https://github.com/bagustris/deep_mlp_ser by Bagus Tris Atmaja, Masato Akagi
    # Computes the metrics CCC, PCC, and RMSE between the sequences x and y
    #  CCC:  Concordance correlation coeffient
    #  PCC:  Pearson's correlation coeffient
    #  RMSE: Root mean squared error
    # Input:  x,y: numpy arrays (one-dimensional)
    # Output: CCC,PCC,RMSE
    
    x_mean = np.nanmean(x)
    y_mean = np.nanmean(y)
    
    covariance = np.nanmean((x-x_mean)*(y-y_mean))
    
    x_var = 1.0 / (len(x)-1) * np.nansum((x-x_mean)**2) # Make it consistent with Matlab's nanvar (division by len(x)-1, not len(x)))
    y_var = 1.0 / (len(y)-1) * np.nansum((y-y_mean)**2)
    
    CCC = (2*covariance) / (x_var + y_var + (x_mean-y_mean)**2)
    
    x_std = np.sqrt(x_var)
    y_std = np.sqrt(y_var)
    
    PCC = covariance / (x_std * y_std)
    
    RMSE = np.sqrt(np.nanmean((x - y)**2))
    
    scores = np.array([ CCC, PCC, RMSE ])
    
    return scores

def load():
    x_test = np.load("sets/x_test.npy")
    x_train = np.load("sets/x_train.npy")
    x_val = np.load("sets/x_val.npy")
    y_test = np.load("sets/y_test.npy")
    y_train = np.load("sets/y_train.npy")
    y_val = np.load("sets/y_val.npy")
    return x_test, x_train, x_val, y_test, y_train, y_val

def getCCC(y_true, y_pre):
    ccc = []
    for i in range(0, 3):
        ccc_, _, _ = calc_scores(y_pre[:, i], y_true[:, i])
        ccc.append(ccc_)
    return ccc  

if __name__=="__main__":
    x_test, x_train, x_val, y_test, y_train, y_val = load()

    wrapper = MultiOutputRegressor(NuSVR(nu=1,kernel='rbf',gamma='scale',C=172600))
    wrapper.fit(x_train, y_train)
    y_pre = wrapper.predict(x_val)
    print(getCCC(y_val, y_pre))
    y_pre = wrapper.predict(x_test)
    print(getCCC(y_test, y_pre))