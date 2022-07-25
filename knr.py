import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets import make_regression
from csv import writer


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

def CCC(y, predict):
    ccc = []
    for i in range(0, 3):
        ccc_, _, _ = calc_scores(predict[:, i], y[:, i])
        ccc.append(ccc_)
    return ccc 

def load():
  train_x = np.load("sets/x_train.npy")
  train_y = np.load("sets/y_train.npy")
  
  val_x = np.load("sets/x_val.npy")
  val_y = np.load("sets/y_val.npy")

  test_x = np.load("sets/x_test.npy")
  test_y = np.load("sets/y_test.npy")

  return train_x, train_y, val_x, val_y, test_x, test_y
    

def knnMultiple():
  train_x, train_y, val_x, val_y, test_x, test_y = load()
  
  model = KNeighborsRegressor(n_neighbors=8, weights="distance", algorithm="auto", p=1, metric="minkowski")          
  model.fit(train_x, train_y)
            
  val_predict = model.predict(val_x)            
  print(CCC(val_y, val_predict))
  
  y_predict = model.predict(test_x)
  print(CCC(test_y, y_predict))
  

knnMultiple()
