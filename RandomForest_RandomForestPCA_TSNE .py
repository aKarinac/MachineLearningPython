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
from sklearn.decomposition import PCA
from sklearn.utils import resample
from sklearn.manifold import TSNE
from imblearn.under_sampling import RandomUnderSampler

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

    
def Random_Forest_Classification():
    x=datele_noastre[["PAY_0","PAY_2","PAY_3","PAY_5"]].to_numpy().reshape(-1, 4)
    y=datele_noastre["default payment next month"].to_numpy()
    
    #x = PolynomialFeatures(degree=2, include_bias=True).fit_transform(x)
    rus = RandomUnderSampler(random_state=42)
    x,y = rus.fit_resample(x, y)
    
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    clf = Pipeline ([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators = 100))
    ]).fit(X_train, y_train) 

    
    y_pred = clf.predict(X_test)
    
    mse = np.mean(y_pred-y_test)
    

    print("matthews_corrcoef: ", matthews_corrcoef(y_test, y_pred))
    print("Confusion matrix: ")
    print(confusion_matrix(y_test, y_pred))
    print()
    print("Classification report: ")
    print(classification_report(y_test, y_pred))
    print()

def Random_Forest_Classification_PCA():
    x=datele_noastre[["PAY_0","PAY_2","PAY_3","PAY_5"]].to_numpy().reshape(-1, 4)
    y=datele_noastre["default payment next month"].to_numpy()

    rus = RandomUnderSampler(random_state=42)
    x,y = rus.fit_resample(x, y)
    print(datele_noastre.shape)
    #x = PolynomialFeatures(degree=3, include_bias=True).fit_transform(x)
    
    pca = PCA(n_components = 2)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

    X_train = pca.fit_transform(X_train)
    
    clf = Pipeline ([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators = 100))
    ]).fit(X_train, y_train) 

    
    X_test = pca.transform(X_test)
    y_pred = clf.predict(X_test)
    
    mse = np.mean(y_pred-y_test)

    print("matthews_corrcoef: ", matthews_corrcoef(y_test, y_pred))
    print("Confusion matrix: ")
    print(confusion_matrix(y_test, y_pred))
    print()
    print("Classification report: ")
    print(classification_report(y_test, y_pred))
    print()
    
    principalDf = pd.DataFrame(data = X_train
             , columns = ['principal_component_1', 'principal_component_2'])
    
    print(principalDf)
    print(datele_noastre['default payment next month'])
    
    finalDf = pd.concat([principalDf, datele_noastre[['default payment next month']]], axis = 1)

    finalDf = finalDf.dropna()
    print(finalDf)
    
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('2 component PCA', fontsize = 20)
    
    targets = [0,1]
    colors = ['r', 'g']
    
    for target, color in zip(targets,colors):
        indicesToKeep = finalDf['default payment next month'] == target
        ax.scatter( finalDf.loc[indicesToKeep, 'principal_component_1']
                   ,finalDf.loc[indicesToKeep, 'principal_component_2']
                   , c = color
                   , s = 50)
    ax.legend(targets)
    ax.grid()

def Random_Forest_Classification_T_SNE():
    x=datele_noastre[["PAY_0","PAY_2","PAY_3","PAY_5"]].to_numpy().reshape(-1, 4)
    y=datele_noastre["default payment next month"].to_numpy()
    
    rus = RandomUnderSampler(random_state=42)
    x,y = rus.fit_resample(x, y)
    print (datele_noastre.shape)
    #x = PolynomialFeatures(degree=2, include_bias=True).fit_transform(x)
    x = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(x)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    clf = Pipeline ([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators = 100))
    ]).fit(X_train, y_train) 

    
    y_pred = clf.predict(X_test)
    
    mse = np.mean(y_pred-y_test)
    

    print("matthews_corrcoef: ", matthews_corrcoef(y_test, y_pred))
    print("Confusion matrix: ")
    print(confusion_matrix(y_test, y_pred))
    print()
    print("Classification report: ")
    print(classification_report(y_test, y_pred))
    print()
    
    principalDf = pd.DataFrame(data = X_train
             , columns = ['principal_component_1', 'principal_component_2'])
    
    print(principalDf)
    print(datele_noastre['default payment next month'])
    
    finalDf = pd.concat([principalDf, datele_noastre[['default payment next month']]], axis = 1)

    finalDf = finalDf.dropna()
    print(finalDf)
    
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('2 component PCA', fontsize = 20)
    
    targets = [0,1]
    colors = ['r', 'g']
    
    for target, color in zip(targets,colors):
        indicesToKeep = finalDf['default payment next month'] == target
        ax.scatter( finalDf.loc[indicesToKeep, 'principal_component_1']
                   ,finalDf.loc[indicesToKeep, 'principal_component_2']
                   , c = color
                   , s = 50)
    ax.legend(targets)
    ax.grid()
    
    
print(curatareDate())    
print(vizualizareDate())

print('Random forest classification: ')
print(Random_Forest_Classification())
print('\n')

print('Random forest classification with PCA: ')
#print(Random_Forest_Classification_PCA())
print('\n')

print('Random forest classification with T-SNE: ')
print(Random_Forest_Classification_T_SNE())
print('\n')