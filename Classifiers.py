# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 21:39:38 2020

@author: Phil
"""
import numpy as np
import pandas as pd

 
from sklearn import svm
from sklearn.svm import SVC
from sklearn.svm import NuSVC
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier

 from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve

import matplotlib.pyplot as plt

''' Load Features '''
features = pd.read_csv('C:/Ptuxiakh/Ektelesh Python/Formatted Features (no PoZ).csv')

'''Remove Files that are missing 40% of data or more'''
PoZ = 0.4
features.drop(features[features.PoZ >= PoZ].index , inplace = True )
features.reset_index(drop = True, inplace= True)
Y = pd.DataFrame(features.Class)
X = pd.DataFrame(features.drop(columns = 'Class'))
del PoZ 

'''Feature Handling'''
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)

''' OverSample the data to make up for class imbalance'''
from imblearn.over_sampling import SMOTE, ADASYN  
print('Percentage of files that belong to class Pre-ictal according to the whole dataset is: %f'   %(Y.sum()/Y.shape[0]) )
X_resampled, Y_resampled = SMOTE().fit_resample(X,Y)


'''Splitting the datasets'''
from sklearn.model_selection import train_test_split
X_train , X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state =42)
#X_train , X_test, y_train, y_test = train_test_split(X_resampled, Y_resampled, test_size=0.2, random_state =42)

# Turn the values into an array for feeding the classification algorithms.
X_train = X_train.values
X_test = X_test.values
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

count = 0 
TS=np.zeros([1,1])

parameters = { 'C': [ 0.1 , 10 , 100, 1000], 'kernel': ('rbf', 'poly')} # C= 1000, kernel rbf
grid_svc = GridSearchCV( SVC() , parameters)
clf = grid_svc.fit(X_train, y_train)
training_score = cross_val_score(clf, X_train, y_train, cv =10 )
print(training_score)
TS = np.vstack((TS,round(training_score.mean(),2)*100))

y_score=clf.predict(X_test)
APS[count] = average_precision_score(y_test, y_score)

classifiers = {
    "SGDClassifier": linear_model.SGDClassifier(),
    "Support Vector Classifier": SVC(),
    "DecisionTreeClassifier": DecisionTreeClassifier(),
    "Random Forest Classifier": RandomForestClassifier(),
    }

#  SVC
parameters = { 'C': [ 0.1 , 10 , 100, 1000], 'kernel': ('rbf', 'poly')} # C= 1000, kernel rbf
grid_svc = GridSearchCV( SVC() , parameters)
grid_svc.fit(X_train, y_train)
clf = grid_svc.best_estimator_
Test= clf.predict(X_test)

 

disp = plot_precision_recall_curve(clf, X_test,y_test)

from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict

svc_pred = cross_val_predict(clf, X_train, y_train, cv=5,method="decision_function")
svc_fpr, svc_tpr, svc_threshold = roc_curve(y_train, svc_pred)
plt.plot(svc_fpr, svc_tpr, label='Support Vector Classifier Score: {:.4f}'.format(roc_auc_score(y_train, svc_pred)))

'''
GridSearchCV(cv=None, error_score=nan,
             estimator=SVC(C=1.0, break_ties=False, cache_size=200,
                           class_weight=None, coef0=0.0,
                           decision_function_shape='ovr', degree=3,
                           gamma='scale', kernel='rbf', max_iter=-1,
                           probability=False, random_state=None, shrinking=True,
                           tol=0.001, verbose=False),
             iid='deprecated', n_jobs=None,
             param_grid={'C': [0.5, 0.75, 1, 2, 4, 16, 32, 64], 'coef0': [0, 1],
                         'degree': [3, 4, 5, 7], 'kernel': ['rbf', 'poly']},
             pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
             scoring=None, verbose=0)
'''
NuSVC
LinearSVC


# Test Kernel
from sklearn.svm import LinearSVC
clf = LinearSVC()
clf.fit(X_train, y_train)
training_score = cross_val_score(clf , X_train, y_train)
print(round(training_score.mean(),2)*100)

# SVC best estimator
svc = grid_svc.best_estimator_