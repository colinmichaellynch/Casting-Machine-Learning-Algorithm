
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

# Import the dataset: change this every iteration for new datasets
iris = datasets.load_iris()
X = pd.DataFrame(iris.data)
y = pd.DataFrame(iris.target)

# Find the factors of the maximum colony size. We do this as, for now, we are just
# interested in seeing whether the number of groups (connectivity) produces a better
# classifier
def factors(n):
    return set(reduce(list.__add__,
                      ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))


# Divide data into training and test sets
X_train, X_test_final, y_train, y_test_final = train_test_split(X, y, test_size=0.2,
                                                                random_state=SEED, stratify=y)

# number of cross validation folds
k_fold = 10

# decision tree depth, these will be decision stumps
tree_depth = 2

# colony size, or the number of weak learners
N_ANTS = 100

# List of window sizes we will iterate through, which are the factors of colony size
window_list = factors(N_ANTS)
window_list = list(window_list)

# List of learning rates we will test
lr_list = np.linspace(start=.1, stop=5, num=len(window_list))

# Convert datasets into ones that can be indexed later
X_temp = np.array(X_train)
y_temp = np.array(y_train)

# Set cross validation
cv_outer = KFold(n_splits=k_fold, shuffle=True, random_state=SEED)

# Initialize lists
validation_accuracy_vec = list()
bias_vec = list()
var_vec = list()
window_size_vec = list()
lr_vec = list()
time_vec = list()

# Set time and counter to keep track of iterations
counter = 0
t_overall = time.time()
t_start = time.time()

### Run simulations ###

# iterate through validation splits
for train_ix, test_ix in cv_outer.split(X_temp):

    X_train, X_test = X_temp[train_ix, :], X_temp[test_ix, :]
    y_train, y_test = y_temp[train_ix], y_temp[test_ix]
    X_train, y_train = pd.DataFrame(X_train), pd.DataFrame(y_train)
    X_test, y_test = pd.DataFrame(X_test), pd.DataFrame(y_test)

    # We do grid search within each validation split. Here we work through all combinations of lr_list and window_list
    for i in range(len(lr_list)):
        lr = lr_list[i]
        for j in range(len(window_list)):

            counter = counter+1
            window_size = window_list[j]

            t = time.time()
            model = ant_colony_classifier(window_size, N_ANTS, SEED, lr, tree_depth)
            # BUG #2: this functon breaks down when the colony size is greater than the number 
            # of datapoints
            model.fit(X_train, y_train)
            # BUG #3: has large, unnecessary output to console. Is there a way to suppresst this
            # output?
            # Make prediction
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
            window_size_vec.append(window_size)
            validation_accuracy_vec.append(acc)

            # printout proportion of iterations run along with total processing time to 
            # current iteration
            print(counter/(k_fold*len(lr_list)*len(window_list)))
            print(time.time()-t_start)

# To choose the final parameters, we find where the maximum validation accuracy is
# Bug 5: Is this simple max function sufficient to choose the best parameter space after a grid search?
max_acc = max(validation_accuracy_vec)
max_index = validation_accuracy_vec.index(max_acc)
window_size = window_size_vec[max_index]
lr = lr_vec[max_index]

# We use the optimal parameters to give us a final test accuracy for the model
model = ant_colony_classifier(window_size, N_ANTS, SEED, lr, tree_depth)
model.fit(pd.DataFrame(X_temp), pd.DataFrame(y_temp))
y_pred = model.predict(pd.DataFrame(X_test_final))
test_accuracy = accuracy_score(y_test_final, y_pred)

# Final time
t_overall = time.time() - t_overall

# P represents the connectivity of the ensemble given the colony size. A highly connected ensemble is one where the ensemble is purely an adaboost algorithm (window_size = N_ANTS) and P = 1. A disconnected ensemble is one where the weak learners are not connected, so P is close to 0. If we only test window_sizes which are factors of N_ANTS, then P can be understood as the proportion of possible groups. For example, lets say that N_ANTS = 10. The factors of 10 are 1, 2, 5, and 10. If we have a optimal window size of 5 (and thus we have 2 groups of 5), then P = 0.75, as this is larger than or equal to 75% of the possible window size space. If window_size = 1, then P = 0.25, if window_size = 2, then P = 0.5, and finally if window_size = 10 then P = 1.
P = sum(window_size >= np.array(window_list))/len(window_list)

### Export data ###

# Values to be added manually into a final spreadsheet
print('Test Accuracy = ' + str(test_accuracy))
print('Window Size = ' + str(window_size))
print('Learning Rate = ' + str(lr))
print('Overall Time (s) = ' + str(t_overall))
print('P = ' + str(P))

# Output data for further analysis in R
df = {'LearningRate': lr_vec, 'WindowSize': window_size_vec, 'ValidationAccuracy':
      validation_accuracy_vec, 'Bias': bias_vec, 'Variance': var_vec, 'Time': time_vec}
df = pd.DataFrame(df)
df.to_csv('IrisAntClassification.csv')
