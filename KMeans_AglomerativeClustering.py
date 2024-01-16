# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 13:34:58 2023

@author: eu
"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error, matthews_corrcoef
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVR, SVR, SVC
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
#from mlxtend.plotting import plot_decision_regions
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn import tree
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.utils import resample
from imblearn.under_sampling import RandomUnderSampler
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

datele_noastre = pd.read_csv('C:\\Users\\eu\\Desktop\\FMI\\MVMOD\\InvatareAutomata\\defaultcredits.csv')
#datele_noastre = pd.read_csv('D:\\LAPTOP\\Desktop\\DESKTOP\\Ovidius_MASTER\\An1 SEM1\\Invatare Automata\\defaultcredits.csv')

print(datele_noastre['default payment next month'].value_counts())
datele_noastre_majority = datele_noastre[datele_noastre['default payment next month']==0]
datele_noastre_minority = datele_noastre[datele_noastre['default payment next month']==1]
datele_noastre_minority_upsampled = resample(datele_noastre_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=23364,    # to match majority class
                                 random_state=123) # reproducible results
datele_noastre_upsampled = pd.concat([datele_noastre_majority, datele_noastre_minority_upsampled])
print(datele_noastre_upsampled['default payment next month'].value_counts())

def vizualizareDate():
    
    datele_noastre.corr()
    plt.figure(dpi=100)
    plt.title('Analiza corelatiilor')
    sns.heatmap(datele_noastre.corr(), annot=True, lw=1, linecolor='white', cmap='viridis')
    plt.xticks(rotation=50)
    plt.yticks(rotation=30)
    plt.show()

def curatareDate():
    print('CURATARE DATE')
    print("Date lipsa: \n", datele_noastre.isnull())
    print()
    print("Suma date lipsa: \n", datele_noastre.isnull().sum())
    print("Date non unique: \n")
    print(datele_noastre.nunique())
    print()
    print("Data Shape: ", datele_noastre.shape)
    #datele_noastre.dropna(subset=["Age"])
    print()
    datele_noastre.drop('ID', inplace=True, axis=1)
    datele_noastre.drop('LIMIT_BAL', inplace=True, axis=1)
    datele_noastre.drop('SEX', inplace=True, axis=1)
    datele_noastre.drop('EDUCATION', inplace=True, axis=1)
    datele_noastre.drop('MARRIAGE', inplace=True, axis=1)
    datele_noastre.drop('PAY_AMT1', inplace=True, axis=1)
    datele_noastre.drop('PAY_AMT2', inplace=True, axis=1)
    datele_noastre.drop('PAY_AMT3', inplace=True, axis=1)
    datele_noastre.drop('PAY_AMT4', inplace=True, axis=1)
    datele_noastre.drop('PAY_AMT5', inplace=True, axis=1)
    datele_noastre.drop('PAY_AMT6', inplace=True, axis=1)
    datele_noastre.drop('AGE', inplace=True, axis=1)

    
def KMeans_Cluster():
    # x=datele_noastre[["PAY_0","PAY_2","PAY_3","PAY_5"]].to_numpy().reshape(-1, 4)
    # y=datele_noastre["default payment next month"].to_numpy()

    # rus = RandomUnderSampler(random_state=42)
    # x,y = rus.fit_resample(x, y)
    # print(x,y)
    
    df = pd.DataFrame(datele_noastre, columns=['PAY_3','PAY_5'])
    kmeans = KMeans(n_clusters=4).fit(df) 
    centroids = kmeans.cluster_centers_ 
    print(centroids)
    plt.scatter(df['PAY_3'], df['PAY_5'], c=kmeans.labels_.astype(float), s=50, alpha=0.5) 
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
    plt.title("K-Means")
    plt.show()
    
    
def AC_Cluster():
    x=datele_noastre[["PAY_3","PAY_5"]].to_numpy().reshape(-1, 2)
    cluster = AgglomerativeClustering(n_clusters=2, metric='euclidean')
    cluster.fit_predict(x)
    print(cluster.labels_)
    plt.scatter(x[:,0],x[:,1], c=cluster.labels_, cmap='rainbow')
    plt.title("Aglomerative Cluster")
    

def Hierarchical_Cluster():
    x=datele_noastre[["PAY_2", "PAY_3"]].to_numpy()
    linked = linkage(x, 'single')
    labelList = range(1, 11)
    plt.figure(figsize=(10, 7))
    dendrogram (linked,
                orientation='top',
                labels=labelList,
                distance_sort='descending', show_leaf_counts=True)
    plt.show()
    

print(curatareDate())    
print(vizualizareDate())

print('KMeans: ')
print(KMeans_Cluster())
print('\n')

print('AC: ')
#print(AC_Cluster())
print('\n')

print('Hierarchical: ')
#print(Hierarchical_Cluster())
print('\n')
