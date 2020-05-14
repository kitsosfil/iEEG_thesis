# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 19:49:49 2020

@author: Phil
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

# Pre-Proccess Extracted Features
features = pd.read_csv('C:/Ptuxiakh/Ektelesh Python/All_Features.csv')
features.drop(columns = 'Unnamed: 0', inplace = True)

# Remove data with Zeros  DONT FORGET ENERGY IS MISSING  [mean(abs(diff()))]
PoZ = 0.4
features.drop(features[features.PoZ >= PoZ].index , inplace = True )
features.reset_index(drop = True, inplace= True)

# str obj to lists...
test=pd.DataFrame()
feature = ['Variance', 'Mobility','Complexity', 'Max_FFT_AR', 'Skewness', 'Kurtosis' ]
count=0
m , n = features.shape
for f in feature:
    new_feature=np.zeros((m,16))
    newfile = 0
    for index,row in features.iterrows():
        x=row[f]
        strx= x.replace("[","").replace("]","").replace("\n","")
        if (    (f == 'Mobility') or (f=='Complexity') ):
            splitx=strx.rsplit(sep = ",")
        else:
            splitx = strx.rsplit(sep =" ") 
        
        new_feature[newfile,:]=np.genfromtxt(splitx)
        newfile = newfile+1 
    for i in range(0,16):
        test.insert(count, f+'_'+str(i+1), new_feature[:,i],True)
        count=count+1

m, n= test.shape       
test.insert(n,'MCV',features.Mean_Channel_Variance,True)
test.insert(n+1,'Welch',features.Welch,True)
test.insert(n+2,'Class',features.Class, True)
#test.insert(99, 'Filename' , features.Filename, True)
Class = pd.DataFrame(features.Class)
del newfile, m, n, count, new_feature, x, strx, splitx, i ,index, row, f



X = pd.DataFrame(test.drop(columns = 'Class'))
Y = Class
# Oversampling Features
from imblearn.over_sampling import SMOTE, ADASYN  
X_resampled, Y_resampled = SMOTE().fit_resample(X,Y)

# TESTING CLASSIFIERS 
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

X_train , X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state =42)
#X_train , X_test, y_train, y_test = train_test_split(X_resampled, Y_resampled, test_size=0.2, random_state =42)
X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
y_test = y_test.values 
 
clf =svm.SVC(kernel='rbf', degree=3, verbose = True)
model = clf.fit(X_train, y_train.ravel() )
training_score = cross_val_score(model , X_train, y_train.ravel())
print(round(training_score.mean(),2)*100)

# Turn the values into an array for feeding the classification algorithms.
X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
y_test = y_test.values 


# Let's implement simple classifiers
classifiers = {
    "LogisiticRegression": LogisticRegression(),
    "KNearest": KNeighborsClassifier(),
    "Support Vector Classifier": SVC(),
    "DecisionTreeClassifier": DecisionTreeClassifier()
}
from sklearn.model_selection import cross_val_score
for key , classifier in classifiers.items():
    classifier.fit(X_train, y_train)
    training_score = cross_val_score(classifier, X_train, y_train, cv=5)
    print("Classifiers: ", classifier.__class__.__name__, "Has  a training score of ", round(training_score.mean(),2 )*100, "% accuracy score")


# Dimensionality Reduction and Clustering

from sklearn.manifold import TSNE 
X_reduced_tsne = TSNE(n_components = 2, random_state = 42).fit_transform(X.values)

from sklearn.decomposition import PCA 
X_reduced_pca = PCA(n_components = 2, random_state = 42).fit_transform(X.values) 

from sklearn.decomposition import TruncatedSVD
X_reduced_svd = TruncatedSVD(n_components = 2, algorithm = 'randomized', random_state = 42).fit_transform(X.values)

'''
Color Testing Codes
df=pd.DataFrame(X_reduced_tsne)
df.rename()
df.insert(2,"Class", Y, True)
colors =np.where(df['Class']==1,'r','b')
df.plot.scatter(x=0, y=1,c=colors)

f, ax = plt.subplots(figsize = (10,10))
ax.scatter(X_reduced_tsne[:,0],X_reduced_tsne[:,1], c = colors, cmap='coolwarm', label ='Inter_Ictal',linewidths = 2)
'''

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
blue_patch = mpatches.Patch(color='#0A0AFF', label='Inter_Ictal')
red_patch = mpatches.Patch(color='#AF0000', label='Pre_Ictal')

f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24,6))
# labels = ['No Fraud', 'Fraud']
f.suptitle('Clusters using Dimensionality Reduction', fontsize=14)

color =np.where(df['Class']==1,'r','b')

# t-SNE scatter plot
ax1.scatter(X_reduced_tsne[:,0], X_reduced_tsne[:,1], c=color, cmap='coolwarm', label='Inter_Ictal', linewidths=2)
ax1.scatter(X_reduced_tsne[:,0], X_reduced_tsne[:,1], c=color, cmap='coolwarm', label='Pre_Ictal', linewidths=2)
ax1.set_title('t-SNE', fontsize=14)

ax1.grid(True)

ax1.legend(handles=[blue_patch, red_patch])


# PCA scatter plot
ax2.scatter(X_reduced_pca[:,0], X_reduced_pca[:,1], c=color, cmap='coolwarm', label='Inter_Ictal', linewidths=2)
ax2.scatter(X_reduced_pca[:,0], X_reduced_pca[:,1], c=color, cmap='coolwarm', label='Pre_Ictal', linewidths=2)
ax2.set_title('PCA', fontsize=14)

ax2.grid(True)

ax2.legend(handles=[blue_patch, red_patch])

# TruncatedSVD scatter plot
ax3.scatter(X_reduced_svd[:,0], X_reduced_svd[:,1], c=color, cmap='coolwarm', label='Inter_Ictal', linewidths=2)
ax3.scatter(X_reduced_svd[:,0], X_reduced_svd[:,1], c=color, cmap='coolwarm', label='Pre_Ictal', linewidths=2)
ax3.set_title('Truncated SVD', fontsize=14)

ax3.grid(True)

ax3.legend(handles=[blue_patch, red_patch])

plt.show()

# Unbalanced Feature's Correlation Matrix
fig, (ax1,ax2) = plt.subplots(2, 1, figsize=(24,20))
sns.heatmap(test.corr(), cmap= 'coolwarm_r', annot_kws={'size':20}, ax=ax1)
ax1.set_title("Imbalanced Correlation Matrix \n (don't use for reference) ", fontsize=14)

# Balanced Feature's Correlation Matrix
over_sampled = pd.DataFrame(X_resampled)
over_sampled.insert(98, 'Class',Y_resampled,True)
sns.heatmap(X_resampled.corr(),  cmap='coolwarm_r', annot_kws={'size':20}, ax=ax2)
ax2.set_title("OverSample Correlation Matrix \n (use for reference)", fontsize=14)
plt.show()

 

# Feature distrubutions 
f, ( (ax1, ax2, ax3, ax4) , (ax5, ax6, ax7, ax8), (ax9, ax10, ax11, ax12), (ax13, ax14, ax15, ax16)) = plt.subplots(4, 4, figsize=(24,24))

'''feature = ['Variance','Mobility','Complexity','Max_FFT_AR','Skewness','Kurtosis' ]'''
feat =feature[1]


'''Inter=0 - Pre=1 Class'''
IPC=0 
cl = features.index[features.Class == IPC].tolist()

feat1 =test[feat+'_1'].loc[cl].values
sns.distplot(feat1, ax=ax1, fit=norm)

feat2 =test[feat+'_2'].loc[cl].values
sns.distplot(feat2, ax=ax2, fit=norm)

feat3 =test[feat+'_3'].loc[cl].values
sns.distplot(feat3, ax=ax3, fit=norm)

feat4 =test[feat+'_4'].loc[cl].values
sns.distplot(feat4, ax=ax4, fit=norm)

feat5 =test[feat+'_5'].loc[cl].values
sns.distplot(feat5, ax=ax5, fit=norm)

feat6 =test[feat+'_6'].loc[cl].values
sns.distplot(feat6, ax=ax6, fit=norm)

feat7 =test[feat+'_7'].loc[cl].values
sns.distplot(feat7, ax=ax7, fit=norm)

feat8 =test[feat+'_8'].loc[cl].values
sns.distplot(feat8, ax=ax8, fit=norm)

feat9 =test[feat+'_9'].loc[cl].values
sns.distplot(feat9, ax=ax9, fit=norm)

feat10 =test[feat+'_10'].loc[cl].values
sns.distplot(feat10, ax=ax10, fit=norm)

feat11 =test[feat+'_11'].loc[cl].values
sns.distplot(feat11, ax=ax11, fit=norm)

feat12 =test[feat+'_12'].loc[cl].values
sns.distplot(feat12, ax=ax12, fit=norm)

feat13 =test[feat+'_13'].loc[cl].values
sns.distplot(feat13, ax=ax13, fit=norm)

feat14 =test[feat+'_14'].loc[cl].values
sns.distplot(feat14, ax=ax14, fit=norm)

feat15 =test[feat+'_15'].loc[cl].values
sns.distplot(feat15, ax=ax15, fit=norm)

feat16 =test[feat+'_16'].loc[cl].values
sns.distplot(feat16, ax=ax16, fit=norm)

plt.show()

test.insert(98,'Class',features.Class, True
test.[feature[1]].loc[test['Class']==1].values

