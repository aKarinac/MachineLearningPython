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
from sklearn.naive_bayes import GaussianNB

#datele_noastre = pd.read_csv('C:\\Users\\alexi\\OneDrive\\Desktop\\MASTER\\AN 1 SEM 1\\IA\\defaultcredits.csv')
datele_noastre = pd.read_csv('C:\\Users\\eu\\Desktop\\FMI\\MVMOD\\InvatareAutomata\\defaultcredits.csv')

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
    
    
    
def Gaussian():
    x=datele_noastre[["BILL_AMT1","BILL_AMT2","BILL_AMT3","BILL_AMT4"]].to_numpy().reshape(-1, 4)
    y=datele_noastre["default payment next month"].to_numpy()
    
    x = PolynomialFeatures(degree=3, include_bias=True).fit_transform(x)
    
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    
    GAUSS = Pipeline ([
        ("scaler", StandardScaler()),
        ("GAUSS", GaussianNB())
    ]).fit(X_train, y_train) 
    
    y_pred = GAUSS.predict(X_test)
    
    r_sq = GAUSS.score(x, y)
    
    print('R^2:', r_sq)
    print("matthews_corrcoef: ", matthews_corrcoef(y_test, y_pred))
    print("Confusion matrix: ")
    print(confusion_matrix(y_test, y_pred))
    print()
    print("Classification report: ")
    print(classification_report(y_test, y_pred))
    print()


print(curatareDate())    
print(vizualizareDate())

print('Gaussian Naive Bayes: ')
print(Gaussian())
print('\n')

