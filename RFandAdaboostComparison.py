
from sklearn import datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from mlxtend.evaluate import bias_variance_decomp
import matplotlib.pyplot as plt
import warnings
from sklearn.ensemble import AdaBoostClassifier
from ant_ensemble_class import *
from functools import reduce
from sklearn.model_selection import KFold
import time

# Set seed for reproducibility
SEED = 520

# Ignore warnings
warnings.filterwarnings("ignore")

# Import the dataset
iris = datasets.load_iris()
X = pd.DataFrame(iris.data)
y = pd.DataFrame(iris.target)

# Divide data into training and test sets
X_train, X_test_final, y_train, y_test_final = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)

# number of cross validation folds
k_fold = 10

# decision tree depth, these will be decision stumps
tree_depth = 2

# colony size, or the number of weak learners
N_ANTS = 100

# List of learning rates we will test
lr_list = np.linspace(start=.1, stop=5, num=9)

# Convert datasets into ones that can be indexed later
X_temp = np.array(X_train)
y_temp = np.array(y_train)

# Set cross validation
cv_outer = KFold(n_splits=k_fold, shuffle=True, random_state=SEED)

# Initialize lists
validation_accuracy_vec = list()
bias_vec = list()
var_vec = list()
lr_vec = list()
time_vec = list()
label_vec = list()

# Set time and counter to keep track of iterations
counter = 0
t_overall = time.time()
t_start = time.time()

#run simulations for adaboost
for train_ix, test_ix in cv_outer.split(X_temp):

    X_train, X_test = X_temp[train_ix, :], X_temp[test_ix, :]
    y_train, y_test = y_temp[train_ix], y_temp[test_ix]
    X_train, y_train = pd.DataFrame(X_train), pd.DataFrame(y_train)
    X_test, y_test = pd.DataFrame(X_test), pd.DataFrame(y_test)
    
    for i in range(len(lr_list)):
        lr = lr_list[i]
        counter = counter+1
        t = time.time()
        model = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=tree_depth,
        max_features="auto"), n_estimators=N_ANTS, random_state=SEED, learning_rate = lr)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        # measure validation accuracy
        acc = accuracy_score(y_test, y_pred)
        # Measure bias and variance
        avg_expected_loss, avg_bias, avg_var = bias_variance_decomp(
            model, X_train.values, y_train.values, X_test.values, y_test.values, 
            loss='0-1_loss', random_seed=SEED, num_rounds=50)

        # Add values to lists
        time_vec.append(time.time() - t)
        bias_vec.append(avg_bias)
        var_vec.append(avg_var)
        lr_vec.append(lr)
        validation_accuracy_vec.append(acc)
        label_vec.append("Adaboost")

        # printout proportion of iterations run along with total processing time to 
        # current iteration
        print(counter/(k_fold*len(lr_list)))
                       
# Set time and counter to keep track of iterations
counter = 0
t_overall = time.time()
t_start = time.time()

#run simulations for random forest
for train_ix, test_ix in cv_outer.split(X_temp):

    X_train, X_test = X_temp[train_ix, :], X_temp[test_ix, :]
    y_train, y_test = y_temp[train_ix], y_temp[test_ix]
    X_train, y_train = pd.DataFrame(X_train), pd.DataFrame(y_train)
    X_test, y_test = pd.DataFrame(X_test), pd.DataFrame(y_test)
    
    for i in range(len(lr_list)):
        lr = lr_list[i]
        counter = counter+1
        t = time.time()
        model = RandomForestClassifier(n_estimators=N_ANTS, random_state=SEED, max_depth = tree_depth)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        # measure validation accuracy
        acc = accuracy_score(y_test, y_pred)
        # Measure bias and variance
        avg_expected_loss, avg_bias, avg_var = bias_variance_decomp(
            model, X_train.values, y_train.values, X_test.values, y_test.values, 
            loss='0-1_loss', random_seed=SEED, num_rounds=50)

        # Add values to lists
        time_vec.append(time.time() - t)
        bias_vec.append(avg_bias)
        var_vec.append(avg_var)
        lr_vec.append(lr)
        validation_accuracy_vec.append(acc)
        label_vec.append("RandomForest")

        # printout proportion of iterations run along with total processing time to 
        # current iteration
        print(counter/(k_fold*len(lr_list)))
        
# Output data for further analysis in R
df = {'ValidationAccuracy': validation_accuracy_vec, 'Bias': bias_vec, 'Variance': var_vec, 'Time': time_vec, 'Model': label_vec}
df = pd.DataFrame(df)
df.to_csv('RFandAdaboost.csv')