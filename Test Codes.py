# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 17:30:47 2020

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

 
#Reading the csv using csv module
import csv
fields = []
rows = []
filenames = []
with open(filename) as csvfile:
    csvreader = csv.reader(csvfile, delimiter=',')
    for row in csvreader:
        filenames.append(row[0])
        directory + Patient[index] + "\\train_" + Patient[index] + "\" + csvfiles.loc[index]['image']

# Power Spectrum Welch
fs=400 
for i in range(1,16):
    channel=data[:,i]
    f , Pxx_den = signal.welch(channel,fs,nperseg=128)
    plt.semilogy(f, Pxx_den) 
 


# Save DataFrame to CSV
path =r'C:\...\'
or path = 'C:\\...\\'
test.to_csv(path+'Formated Features', index = False)

#Print all columns 
    print(features[['Filename']])
    
#Find a row while searching for a value in dataframe
    features.loc[features['Filename'] == '1_10_0.mat']
    
#Find index of a row
    features.index[features['Filename'] == '1_10_0.mat'].tolist()

#To add a new row we use loc or iloc to insert a new row
    m,n = df.shape
    features.loc[m] = []
    
#Find if a file exists in the DataFrame
    flag = (features['Filename'] == dataname)
    sum( flag == True)
    #if found 1 or else 0 
    
# Remove data with Zeros  DONT FORGET ENERGY IS MISSING  [mean(abs(diff()))]
PoZ = 0.4
features.drop(features[features.PoZ >= PoZ].index , inplace = True )
features.reset_index(drop = True, inplace= True)

# str obj to np array   
    x= features.loc[1]['Variance']
    x.replace("[","",100)
    x.replace("]","",100)
    x.replace("\n","",100)
    listx= x.replace("[","").replace("]","").replace("\n","")
    x.split
    np.genfromtxt(x.replace("]","",1).replace("[","",1).replace("\n","",10))
    
#Split Patient DataSets 
    Patient_1 = pd.DataFrame(features[features.Patient == 1])
    Patient_2 = pd.DataFrame(features[features.Patient == 2])
    Patient_3 = pd.DataFrame(features[features.Patient == 3])

#Split Classes
    Patient_1_Inter = pd.DataFrame(Patient_1[Patient_1.Class == 0])
    Patient_2_Inter = pd.DataFrame(Patient_2[Patient_2.Class == 0])
    Patient_3_Inter = pd.DataFrame(Patient_3[Patient_3.Class == 0])
    
    Patient_1_Pre = pd.DataFrame(Patient_1[Patient_1.Class == 1])
    Patient_2_Pre = pd.DataFrame(Patient_2[Patient_2.Class == 1])
    Patient_3_Pre = pd.DataFrame(Patient_3[Patient_3.Class == 1])