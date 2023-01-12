#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 17:51:11 2019
 
@author: colinmichaellynch
"""
 
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from sklearn import datasets
 
iris = datasets.load_diabetes()
X = pd.DataFrame(iris.data[:, :10])  # we only take the first two features.
y = pd.DataFrame(np.log10(iris.target))
 
SEED = 520
 
#X = np.random.rand(100,5)
#X = pd.DataFrame(X)
#y = np.zeros([100,1],int)
#y[int(y.shape[0]/3)+1:int(2*y.shape[0]/3)] = 1
#y[int(2*y.shape[0]/3)+1:] = 2
#y = pd.DataFrame(y)
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
 
#Window size
window_size = 11
#Number of ants
N_ANTS = 1000
#learning rate
lr = .01
#tree depth
tree_depth = 2
 
 
#Creates classifiers that represent the colony (group and remainder group)
def ant_colony_regressor(window_size,N_ANTS,SEED,lr,tree_depth):
    group = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth=tree_depth),n_estimators=window_size,learning_rate=lr,random_state=SEED,loss='square')
    A = BaggingRegressor(base_estimator=group,n_estimators=int(N_ANTS/window_size),max_samples=(window_size/N_ANTS),bootstrap=True,oob_score=True,n_jobs=-1,random_state=SEED,verbose=True)
    B = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth=tree_depth),n_estimators=N_ANTS%window_size,learning_rate=lr,random_state=SEED,loss='square')
    regressors = [A,B]
    return regressors
 
#trains the classifiers
def fit(reg,X,y,window_size,N_ANTS,SEED):
    Sample_X = X.sample(frac=window_size/N_ANTS,replace=True,random_state=SEED)
    Sample_y = y.loc[Sample_X.index]
    reg[0].fit(X_train,y_train)
    reg[1].fit(Sample_X,Sample_y)
    return reg
 
#predicts classes
def predic(reg,X,window_size,N_ANTS):
    pa = reg[0].predict(X)
    pb = reg[1].predict(X)
    pa = ((pa * int(N_ANTS / window_size)) + pb) / (int(N_ANTS / window_size) + 1)
    #i = 0
    #for col in range(proba.shape[1]):
    #    if col in Sample_y.values:
    #        proba[:, col] = (proba[:, col] * int(N_ANTS / window_size) + probb[:, i]) / (
    #                    int(N_ANTS / window_size) + 1)
    #        i += 1
    #    else:
    #        proba[:, col] = proba[:, col] * int(N_ANTS / window_size) / (int(N_ANTS / window_size) + 1)
    return pa
 
 
 
eft = ant_colony_regressor(window_size,N_ANTS,SEED,lr,tree_depth)
eft = fit(eft,X_train,y_train,window_size,N_ANTS,SEED)
pred_price = predic(eft,X_test,window_size,N_ANTS)
 
score = r2_score(y_test,pred_price)
#score = np.sqrt(mean_squared_error(y_test,pred_price))
#pred = np.argmax(prob,axis=1)
#score = accuracy_score(y_test,pred)
print(score)
 

#comparison to RandomForest Regressor
from sklearn.ensemble import RandomForestRegressor
 
regr = RandomForestRegressor(max_depth=None, random_state=SEED, n_estimators=50)
regr.fit(X_train, y_train)
pr = regr.predict(X_test)
score = r2_score(y_test,pr)
print(score)
