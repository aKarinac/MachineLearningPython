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
    
    
    
def Linear_SVR():
    x=datele_noastre["PAY_4"].to_numpy().reshape(-1, 1)
    y=datele_noastre["PAY_5"].to_numpy()
    
    svm_reg = LinearSVR(epsilon=1.5).fit(x, y) 
    y_pred = svm_reg.predict(x)
    #X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    r_sq = svm_reg.score(x, y)
    mse = mean_squared_error(y, y_pred)

    print('R^2:', r_sq)
    print('MSE:', mse)
    print('Intercept:', svm_reg.intercept_)
    print('Slope:', svm_reg.coef_)

    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='r')
    plt.plot(x, y_pred, color='b')
    plt.title(f'SVR liniar: R^2 = {r_sq}')
    plt.show()
    
    # plt.figure(dpi=100)
    # plot_decision_regions(X = X_test, y = y_test.values, model = svm_reg)
    # plt.title("SVR with linear kernel")
    # plt.show()

def Poli_SVR():
    x = datele_noastre["BILL_AMT2"].to_numpy().reshape(-1, 1)
    y = datele_noastre["BILL_AMT1"].to_numpy()
    
    x_poly = PolynomialFeatures(degree=2, include_bias=True).fit_transform(x)

    svm_reg = Pipeline([
        ("scaler", StandardScaler()),
        ("poly_svr", SVR(kernel="poly", degree=4, C=100, coef0=1)) 
    ])

    svm_reg.fit(x_poly, y) 

    y_pred = svm_reg.predict(x_poly)

    r_sq = svm_reg.score(x_poly, y)
    mse = mean_squared_error(y, y_pred)

    print('R^2:', r_sq)
    print('Mean Squared Error:', mse)
    print('Intercept:', svm_reg["poly_svr"].intercept_)

    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='r')
    plt.plot(x, y_pred, color='b')
    plt.title(f'Polynomial SVR: R^2 = {r_sq}')
    plt.show()
    # x_poly = PolynomialFeatures(degree=2, include_bias=True).fit_transform(x)
    
    # svm_reg = SVR(kernel="poly", degree=2, C=100, epsilon=0.1).fit(x_poly, y) 

    # y_pred = svm_reg.predict(x_poly)

    # r_sq = svm_reg.score(x_poly, y)
    # mse = mean_squared_error(y, y_pred)

    # print('R^2:', r_sq)
    # print('Mean Squared Error:', mse)
    # print('Intercept:', svm_reg.intercept_)

    # plt.figure(figsize=(10, 6))
    # plt.scatter(x, y, color='b')
    # plt.plot(x, y_pred, color='r')
    # plt.title(f'SVR polinomial: R^2 = {r_sq}')
    # plt.show()

def knn():
    x=datele_noastre[["BILL_AMT1","BILL_AMT2","BILL_AMT3","BILL_AMT4"]].to_numpy().reshape(-1, 4)
    y=datele_noastre["default payment next month"].to_numpy()
    
    x = PolynomialFeatures(degree=3, include_bias=True).fit_transform(x)

    KNN = Pipeline ([
        ("scaler", StandardScaler()),
        ("KNN", KNeighborsClassifier(n_neighbors=3))
    ]).fit(x, y)

    y_pred = KNN.predict(x)
    
    print("R^2: ", KNN["KNN"].score(x, y))
    print("matthews_corrcoef: ", matthews_corrcoef(y, y_pred))
    print()
    print("Confusion matrix: ")
    print(confusion_matrix(y, y_pred))
    print()
    print("Classification report: ")
    print(classification_report(y, y_pred))
    print()


def svc():
    x=datele_noastre[["BILL_AMT1","BILL_AMT2","BILL_AMT3","BILL_AMT4"]].to_numpy().reshape(-1, 4)
    y=datele_noastre["default payment next month"].to_numpy()
    
    
    x = PolynomialFeatures(degree=2, include_bias=True).fit_transform(x)

    svc_clas = Pipeline ([
        ("scaler", StandardScaler()),
        ("svc_clas", SVC(kernel="poly", degree=2, C=100, coef0=1))
    ]).fit(x,y)

    y_pred = svc_clas.predict(x)

    print("R^2: ", svc_clas["svc_clas"].score(x, y))
    print("matthews_corrcoef: ", matthews_corrcoef(y, y_pred))
    print()
    print("Confusion matrix: ")
    print(confusion_matrix(y, y_pred))
    print()
    print("Classification report: ")
    print(classification_report(y, y_pred))
    print()

    
print(curatareDate())    
print(vizualizareDate())

print('SVR liniar: ')
print(Linear_SVR())
print('\n')

print('SVR polinomial: ')
#print(Poli_SVR())
print('\n')

print('KNN: ')
#print(knn())
print('\n')

print('SVC: ')
#print(svc())
print('\n')

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

# KNN = KNeighborsClassifier(n_neighbors=3).fit(x, y)

# y_pred = KNN.predict(x)

# StdSc = StandardScaler()

# StdSc = StdSc.fit(x)
# x_scaled = KNN.transform(x)

# print("Scor: ", accuracy_score(y_test, y_pred))
