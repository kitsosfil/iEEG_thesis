# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 19:51:46 2020

@author: Phil
"""

#danezis@santowines.gr

import numpy as np
import pandas as pd
import scipy.io as sio 
import statistics
import pyeeg as eeg

from scipy.stats import skew
from scipy.stats import kurtosis
from scipy import signal


filename = "train_and_test_data_labels_safe.csv"
csvfiles = pd.read_csv(filename)

directory = r"C:\Ptuxiakh\TrainData\Patient_"

# Creating the DataFrame that will hold the features
columns=['Filename','Patient','Class','PoZ','Max','Min','Variance','Mean_Channel_Variance','Mobility','Complexity','Max_FFT_AR','Skewness','Kurtosis','Welch']

Features = pd.DataFrame(columns = columns )

for index,row in csvfiles.iterrows():
    
    # File that is Processed
    data_name = csvfiles.loc[index]['image']
    print(data_name)
    
    # Number of Patient
    patient = csvfiles.loc[index]['image'][0]
    #Directory to load file
    data_dir= directory + patient + r'\train_' + patient + r'\\' + csvfiles.loc[index]['image']  
    # Class of file being proccessed 
    data_class =csvfiles.loc[index]['class']
    '''
        Some files where missing in the first pass
        The code above was so that the features of the missing files
        could be loaded in the existing DataFrame
        
    file_exists = (Features['Filename'] == data_name)
    flag = sum( file_exists == True)
    
    if ((csvfiles.loc[index]['safe']==1) and (flag==0)) :
  
        '''
    if (csvfiles.loc[index]['safe']==1):
        try:
            mat_file = sio.loadmat(data_dir, verify_compressed_data_integrity = False)    
            #Converts mat_file to correct format
            #sig = {n: mat['dataStruct'][n][0, 0] for n in mat['dataStruct'].dtype.names}
            data = mat_file['dataStruct']['data'][0,0]
            # Percentage of Zeroes
            PoZ=sum(sum((data==0)==True))/(240000*16)
            # Max Value
            Max=data.max()
            # Min Value
            Min=data.min()
            # Variance
            Variance = np.array(data.var(0))
            # Variance of mean channel
            Mean_Channel_Variance=data.mean(1).var(0)
            # Hjorth Parameters
            Mobility = []
            Complexity = []
            for i in range(0,16):
                first_diff_order = np.diff(data[ :, i])
                mob ,  comp = eeg.hjorth(data[ :, i], first_diff_order.tolist())
                Mobility.append(mob)
                Complexity.append(comp)
                del first_diff_order, mob, comp
                #hjorth.append(eeg.hjorth(data[ :, i], []))   
            # Max FFT Amplitude
            #np.fft.fft2(data).max(0)
            Max_FFT_AR= abs(np.fft.fft(data).max(0))   
            #Skewness
            Skewness=skew(data)
            #Kurtosis
            Kurt=kurtosis(data)
            #Power Spectral frequency Welch's method
            fs=400
            for i in range(0,16):
                channel=data[:,i]
                f , Pxx =signal.welch(channel,fs,nperseg=8, noverlap = 4 ,scaling= 'density',)
                #print(statistics.mean(Pxx*f))
                pwelch = statistics.mean(Pxx * f)
                del f, Pxx, channel

            m, n = Features.shape
            Features.loc[m] = [data_name, patient , data_class, PoZ, Max, Min, Variance, 
                               Mean_Channel_Variance, Mobility, Complexity, Max_FFT_AR, Skewness, Kurt, pwelch]

        except IOError:
            print("Could not load file:" , data_name)
 

