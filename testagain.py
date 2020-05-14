# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 04:08:58 2020

@author: Phil
"""
import numpy as np
import pandas as pd
import scipy.io as sio 
import csv 

import statistics
import pyeeg as eeg

from scipy.stats import skew
from scipy.stats import kurtosis
from scipy import signal

features = pd.read_csv('C:/Users/Phil/Desktop/All_Features.csv')
'''
features.drop(columns = 'Unnamed: 0', inplace = True)
features.rename(columns = {'Unnamed: 0': 'Filename',
                           '0': 'Patient',
                           '1': 'Class', 
                           '2': 'PoZ',
                           '3': 'Max',
                           '4': 'Min',
                           '5': 'Variance',
                           '6': 'Mean_Channel_Variance',
                           '7': 'Mobility',
                           '8': 'Complexity',
                           '9': 'Max_FFT_AR',
                           '10': 'Skewness',
                           '11': 'Kurtosis',
                           '12': 'Welch'}, inplace = True)
'''

columns=['Filename',
         'Patient',
         'Class',
         'PoZ',
         'Max',
         'Min',
         'Variance_1','Variance_2','Variance_3','Variance_4','Variance_5','Variance_6','Variance_7','Variance_8','Variance_9', 'Variance_10','Variance_11', 'Variance_12', 'Variance_13','Variance_14','Variance_15','Variance_16',
         'Mean_Channel_Variance',
         'Mobility_1','Mobility_2','Mobility_3', 'Mobility_4','Mobility_5', 'Mobility_6','Mobility_7', 'Mobility_8','Mobility_9','Mobility_10', 'Mobility_11', 'Mobility_12','Mobility_13','Mobility_14','Mobility_15','Mobility_16',
         'Complexity_1','Complexity_2','Complexity_3','Complexity_4','Complexity_5','Complexity_6','Complexity_7','Complexity_8','Complexity_9','Complexity_10','Complexity_11','Complexity_12','Complexity_13','Complexity_14','Complexity_15','Complexity_16',
         'Max_FFT_AR_1','Max_FFT_AR_2','Max_FFT_AR_3','Max_FFT_AR_4','Max_FFT_AR_5','Max_FFT_AR_6','Max_FFT_AR_7','Max_FFT_AR_8','Max_FFT_AR_9','Max_FFT_AR_10','Max_FFT_AR_11','Max_FFT_AR_12','Max_FFT_AR_13','Max_FFT_AR_14','Max_FFT_AR_15','Max_FFT_AR_16',
         'Skewness_1','Skewness_2','Skewness_3','Skewness_4','Skewness_5','Skewness_6','Skewness_7','Skewness_8','Skewness_9','Skewness_10','Skewness_11','Skewness_12','Skewness_13','Skewness_14','Skewness_15','Skewness_16',
         'Kurtosis_1','Kurtosis_2','Kurtosis_3','Kurtosis_4','Kurtosis_5','Kurtosis_6','Kurtosis_7','Kurtosis_8','Kurtosis_9','Kurtosis_10','Kurtosis_11','Kurtosis_12','Kurtosis_13','Kurtosis_14','Kurtosis_15','Kurtosis_16',
         'Welch']


# Split np.arrays to columns
m, n = features.shape
feature = ['Variance', 'Mobility', 'Complexity', 'Max_FFT_AR', 'Skewness', 'Kurtosis' ]
for file in range(0,m):
    for f in feature:
        x = features.loc[file][f]
        strx = x.replace("[","").replace("]","").replace("\n","")
        splitx = strx.rsplit(sep =" ")
        new_feature = np.genfromtxt(splitx)
        for index in new_feature:
            print(index)
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

# Pre-Proccess Extracted Features
Files = pd.read_csv('C:/Users/Phil/Desktop/All_Features.csv')
features = pd.read_csv('C:/Users/Phil/Desktop/Formatted Features (no PoZ).csv)

PoZ = 0.4
features.drop(Files[Files.PoZ >= PoZ].index , inplace = True )
features.reset_index(drop = True, inplace= True)







