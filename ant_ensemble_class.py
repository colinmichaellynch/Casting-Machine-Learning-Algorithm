from enum import auto
from sklearn.ensemble import BaggingClassifier
#from AdaboostNew import AdaBoostClassifierNew
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd

class ant_colony_classifier:
    # Creates classifiers that represent the colony (group and remainder group)
    def __init__(self,window_size,N_ANTS,SEED,lr,tree_depth):
        self.seed = SEED
        self.window_size = window_size
        self.N_ANTS = N_ANTS
        group = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=tree_depth,
        max_features="auto"),n_estimators=window_size,learning_rate=lr,random_state=SEED)
        A = BaggingClassifier(base_estimator=group,n_estimators=int(N_ANTS/window_size),
        max_samples=1.0,bootstrap=True,oob_score=True,n_jobs=-1,random_state=SEED,verbose=True)
        self.main_group = A
        self.fitted_main_group = None
        self.fitted_remaining_group = None
        self.classes_remaining_group = None
        if N_ANTS%window_size !=0:
            B = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=tree_depth),
            n_estimators=N_ANTS%window_size,learning_rate=lr,random_state=SEED)
            self.remaining_group = B
        else:
            self.remaining_group = None

    # Trains the classifiers
    def fit(self,X,y):
        if isinstance(X,np.ndarray):
            X = pd.DataFrame(X)
            y = pd.DataFrame(y)
        
        Sample_X = X.sample(frac=self.window_size/self.N_ANTS,replace=True,random_state=self.seed)
        Sample_y = y.iloc[Sample_X.index]
        
        # Random subset of features
        n_features  = np.shape(X)[1]
        k = 1 + (self.window_size-1)/(self.N_ANTS-1)
        max_features = (int) (np.round(np.sqrt(n_features**k)))
        self.main_group.max_features = max_features
        # X = X[:, np.random.choice(X.shape[1], max_features, replace=True)]
        # X = X.sample(n=max_features,replace=False,axis=1)


        self.fitted_main_group = self.main_group.fit(X,y)
        self.classes_remaining_group = Sample_y.drop_duplicates()
        
        if self.N_ANTS % self.window_size != 0:
            self.fitted_remaining_group = self.remaining_group.fit(Sample_X,Sample_y)
        
        return self
    
    # Predicts classes
    def predict(self,X):
        proba = self.fitted_main_group.predict_proba(X)
        if self.N_ANTS % self.window_size != 0:
            probb = self.fitted_remaining_group.predict_proba(X)
            i = 0
            for col in range(proba.shape[1]):
                if col in self.classes_remaining_group.values:
                    proba[:, col] = (proba[:, col] * int(self.N_ANTS / self.window_size) + 
                    probb[:, i]) / (int(self.N_ANTS / self.window_size) + 1)
                    i += 1
                else:
                    proba[:, col] = proba[:, col] * int(self.N_ANTS / self.window_size) / (int(self.N_ANTS / self.window_size) + 1)
        # return proba
        return np.argmax(proba, axis=1)

    # # Predicts classes
    # def predict(self,X):
    #     proba = self.fitted_main_group.predict_proba(X)
    #     if self.N_ANTS % self.window_size != 0:
    #         probb = self.fitted_remaining_group.predict_proba(X)
    #         for col in range(proba.shape[1]):
    #             if col in self.classes_remaining_group.values:
    #                 proba[:, col] = (proba[:, col] * int(self.N_ANTS / self.window_size) + probb[:, col]*self.window_size) / self.N_ANTS
    #             else:
    #                 proba[:, col] = proba[:, col] * int(self.N_ANTS / self.window_size) / self.N_ANTS
    #     return proba
    #     #return np.argmax(proba, axis=1)


