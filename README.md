# Casting-Machine-Learning-Algorithm
Implementation and Validation Novel Meta-Algorithm for Ensemble Machine Learning Methods

## Table of Contents

* [Supporting Documentation](https://github.com/colinmichaellynch/Casting-Machine-Learning-Algorithm/blob/main/Casting%20Meta%20Algorithm.docx)

* Data
  - [Classification Results from Casting Algorithm](https://github.com/colinmichaellynch/Casting-Machine-Learning-Algorithm/blob/main/IrisAntClassification.csv)
  - [Classification Results from Reference RF and Adaboost Algorithms](https://github.com/colinmichaellynch/Casting-Machine-Learning-Algorithm/blob/main/RFandAdaboost.csv)

* Code
  - [Casting Function for Classification](https://github.com/colinmichaellynch/Casting-Machine-Learning-Algorithm/blob/main/ant_ensemble_class.py)
  - [Casting Function for Regression](https://github.com/colinmichaellynch/Casting-Machine-Learning-Algorithm/blob/main/ant_ensemble_regression.py)
  - [Testing Casting Algorithm on Iris Dataset Classification](https://github.com/colinmichaellynch/Casting-Machine-Learning-Algorithm/blob/main/ant_ensemble_class.py)
  - [Testing Reference Algorithms (RF/Adaboost) on Iris Dataset Classification](https://github.com/colinmichaellynch/Casting-Machine-Learning-Algorithm/blob/main/RFandAdaboostComparison.py)
  - [Visualization of Results](https://github.com/colinmichaellynch/Casting-Machine-Learning-Algorithm/blob/main/graphTradeoff.R)

## Background

Here we define a novel meta-algorithm for reconstructing serial and parallel ensembles. This meta algorithm - called casting - can be used to create a continuum of base learners which are either loosely connected (i.e. a parallel ensemble) or communicate tightly via a boosting algorithm (i.e. a serial ensemble). While casting can be defined generally to create a network of any set of machine learning algorithms, here we develop the case where each base estimator is a weak learner (a decision stump). This spectrum of learners is anchored at the parallel end by a random forest while its serial counterpart is Adaboost. Casting works by bagging subgroups of stumps (castes) which use Adaboost to communicate internally but operate independently. Casting introduces a new hyperparameter which controls the number of groups. If the hyperparameter P is 0, then all groups of a size of 1 so casting functions as a random forest. If P is 1, then it’s equivalent to Adaboost. If P is set some value in between these two extremes, then we have a novel ensemble which mimics natural systems. In particular, this algorithm was inspired by the observation that colonies of ants tend not to be perfectly connected nor disconnected. They tend to form subgroups, or castes focused on completing a certain type of task. For instance, some polymorphic ants have larger workers that act as sentires for other, smaller workers. 

This design could allow us to optimally trade-off bias and variance. Highly communicative ensembles might be more prone to bias than models with low social adhesions, as the decision of each base estimator depends on the decisions of previous estimators in the chain. Conversely, the decisions of independent networks are likely to be more variable. Intermediate networks may be able to get the best of both worlds, at least for some datasets. Here, we first validate whether or not this algorithm makes the same predictions as a random forest when P = 0 and as Adaboost when P = 1. If this is the case, we then determine whether or not intermediate values of P trade off bias and variance. 

## Methods 

* See [documentation](https://github.com/colinmichaellynch/Casting-Machine-Learning-Algorithm/blob/main/Casting%20Meta%20Algorithm.docx) for the math behind constructing the casting algorithm. 

* 


![variance](https://user-images.githubusercontent.com/61156429/212184677-7e85bd78-d437-462d-ade1-74bd5e48f265.png)
![Bias](https://user-images.githubusercontent.com/61156429/212184679-e9f210c1-f17b-4071-904d-ab4b1d0f2476.png)
![timeelapsed](https://user-images.githubusercontent.com/61156429/212184680-ebb1d98f-1dba-4f43-b430-26fc4f9e75c7.png)
![Rplot](https://user-images.githubusercontent.com/61156429/212184681-61b54ea7-f00d-4a16-bc49-eff56155a80a.png)
