# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 17:16:44 2020

@author: Dell
"""
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib','qt')
 
import pandas as pd
pd.set_option('display.max_columns',200)
pd.set_option('display.max_rows',500)

import numpy as np
import matplotlib.pyplot as plt

import seaborn as sn

dataset = pd.read_csv('train.csv')

# Cleaning Data
# Removing NaNMasVnrArea 
dataset.head()
dataset.isna().any()
dataset.isna().sum()

dataset = dataset.drop(columns = ['MiscFeature','Fence','PoolQC','FireplaceQu','Alley'])

# Taking care of missing data
"""
dataset.fillna(value=dataset['LotFrontage'].mean(),inplace=True)

"""
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer = imputer.fit(dataset.iloc[:,[3,25,57]].values)
dataset[['LotFrontage','MasVnrArea','GarageYrBlt']]=imputer.transform(dataset.iloc[:,[3,25,57]].values).astype('int32')
dataset.fillna(value='None Avialabe',inplace=True)
dataset.isna().sum()

X = dataset.drop(columns='SalePrice') #independent fields
y= dataset['SalePrice'] #label



numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
Xnumeric = X.select_dtypes(include=numerics)

fig = plt.figure(figsize=(15, 12))
plt.suptitle('Histograms of Independent Columns(Continuous values)', fontsize=20)
## Histograms
for i in range(1,Xnumeric.shape[1]):
    plt.subplot(7, 6, i)
    f = plt.gca()
    f.axes.get_yaxis().set_visible(False)
    f.set_title(Xnumeric.columns.values[i])
    vals = np.size(Xnumeric.iloc[:, i ].unique())
    if vals >= 100:
       vals = 100
    plt.hist(Xnumeric.iloc[:, i],bins=vals,color='#3F5D7D')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])




categorical= ['object']
Xcategorical= X.select_dtypes(include=categorical)
fig = plt.figure(figsize=(15, 12))
plt.suptitle('Count Plot of Independent Columns(Categorical)', fontsize=20)
## Bars diagrams
for i in range(1,Xcategorical.shape[1]):
    plt.subplot(7, 6, i)
    f = plt.gca()
    f.axes.get_yaxis().set_visible(False)
    f.set_title(Xcategorical.columns.values[i])
    vals = np.size(Xcategorical.iloc[:, i ].unique())
    if vals >= 100:
       vals = 100
    sn.countplot(Xcategorical.columns.values[i],data=Xcategorical,color='#3F5D7D')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

X['Utilities'].value_counts()  
X['Street'].value_counts()  

X=X.drop(columns=['Utilities'])
dataset=dataset.drop(columns=['Utilities'])
X=X.drop(columns=['Street'])
dataset=dataset.drop(columns=['Street'])

X=X.drop(columns=['Id'])

## Correlation with Predit Sales Price
fig = plt.figure(figsize=(15, 12))
X.corrwith(y).plot.bar(
        title = "Correlation with Predit Sales Price", fontsize = 15,
        rot = 45, grid = True)

#Correlation matrix
sn.set(style="white")

# Compute the correlation matrix
corr = X.corr()
# Generate a custom diverging colormap
cmap = sn.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
fig = plt.figure(figsize=(15, 12))
sn.heatmap(corr,  cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

dataset = pd.get_dummies(dataset,drop_first=True)

dataset.to_csv('new_train_data.csv', index = False)

