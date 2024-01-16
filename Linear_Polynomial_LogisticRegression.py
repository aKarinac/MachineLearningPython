# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 17:16:02 2023

@author: eu
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
#from sklearn.preprocessing import imputer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.metrics import matthews_corrcoef
from sklearn.linear_model import ElasticNet

datele_noastre = pd.read_csv('C:\\Users\\eu\\Desktop\\FMI\\MVMOD\\InvatareAutomata\\defaultcredits.csv')

def vizualizareDate():
    print(datele_noastre.head(10))
    print()
    print(datele_noastre.shape)
    print()
    print(datele_noastre.columns)
    print()
    print(datele_noastre.info)
    print()
    print('DESCRIBE:\n' ,datele_noastre.describe())
    print()
    print(datele_noastre.corr())
    print()
    plt.figure(dpi=100)
    plt.title('Analiza corelatiilor')
    sns.heatmap(datele_noastre.corr(), annot=True, lw=1, linecolor='white', cmap='viridis')
    plt.xticks(rotation=60)
    plt.yticks(rotation=30)
    plt.show()

def curatareDate():
    datele_noastre.isnull()
    datele_noastre.isnull().sum()
    datele_noastre.nunique()
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

def regresieLiniaraS():
    x=datele_noastre["BILL_AMT1"].to_numpy().reshape(-1, 1)
    y=datele_noastre["BILL_AMT2"].to_numpy().reshape(-1, 1)
    
    from sklearn.linear_model import LinearRegression
    model=LinearRegression().fit(x, y)
    y_pred=model.predict(x)
    
    r_sq=model.score(x, y)
    print('coef R^2: ', r_sq)
    print('intercept: ', model.intercept_)
    print('slope: ', model.coef_)
    
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    regressor = LinearRegression()
    regressor.fit(X_train, y_train) # antrenarea algoritmului
    print(regressor.intercept_)
    print(regressor.coef_)
    #compararea valorilor reale cu cele prezise
    y_pred = regressor.predict(X_test)
    #Calcului Erorilor
    print('Eroarea medie absoluta:', metrics.mean_absolute_error(y_test, y_pred)) # MAE
    print('Eroare medie patratica:', metrics.mean_squared_error(y_test, y_pred)) # MSE 
    print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    
    plt.scatter(x, y, color='red')
    plt.plot(X_test, y_pred, color='blue')
    plt.show()

def regresiePoliUni():
    x=datele_noastre["BILL_AMT2"].to_numpy().reshape(-1, 1)
    y=datele_noastre["BILL_AMT1"].to_numpy()

    x_= PolynomialFeatures(degree=3, include_bias=True).fit_transform(x)
    model= LinearRegression().fit(x_, y)

    r_sq=model.score(x_,y)
    intercept, coefficients = model.intercept_, model.coef_
    
    x_range = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
    x_range_poly= PolynomialFeatures(degree=3, include_bias=True).fit_transform(x_range)
    y_pred = model.predict(x_range_poly)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='r')
    plt.plot(x_range, y_pred, color='b')
    plt.show()
    
    X_train, X_test, y_train, y_test = train_test_split(x_, y, test_size=0.3, random_state=0)
    regressor = LinearRegression()
    regressor.fit(X_train, y_train) # antrenarea algoritmului
    print(regressor.intercept_)
    print(regressor.coef_)
    print('coef R^2:', r_sq)
    print('intercept:',intercept)
    print('slope:',coefficients, sep="\n")
    y_pred = regressor.predict(X_test)
  
    
def regresieBivariataS():
    y=datele_noastre["BILL_AMT2"].to_numpy().reshape(-1, 1)
    x=datele_noastre[["BILL_AMT1","BILL_AMT3"]].to_numpy().reshape(-1, 2)
    x1=datele_noastre["BILL_AMT1"].to_numpy().reshape(-1, 1)
    x2=datele_noastre["BILL_AMT3"].to_numpy().reshape(-1, 1)
    model=LinearRegression().fit(x, y)
    y_pred=model.predict(x)
    
    r_sq=model.score(x, y)
    print('coef R^2: ', r_sq)
    print('intercept: ', model.intercept_)
    print('slope: ', model.coef_)
    
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x1, x2, y, color='r')

    x1_range = np.linspace(x1.min(), x1.max(), 100)
    x2_range = np.linspace(x2.min(), x2.max(), 100)
    x1_mesh, x2_mesh = np.meshgrid(x1_range, x2_range)
    y_pred_mesh = model.predict(np.concatenate((x1_mesh.flatten().reshape(-1, 1), x2_mesh.flatten().reshape(-1, 1)), axis=1))
    y_pred_mesh = y_pred_mesh.reshape(x1_mesh.shape)
    
    ax.plot_surface(x1_mesh, x2_mesh, y_pred_mesh, alpha = 0.5)
    
    
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    regressor = LinearRegression()
    regressor.fit(X_train, y_train) # antrenarea algoritmului
    print(regressor.intercept_)
    print(regressor.coef_)
    #compararea valorilor reale cu cele prezise
    y_pred = regressor.predict(X_test)
    #Calcului Erorilor
    print('Eroarea medie absoluta:', metrics.mean_absolute_error(y_test, y_pred)) # MAE
    print('Eroare medie patratica:', metrics.mean_squared_error(y_test, y_pred)) # MSE 
    print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    
  
    
def regresieBivariataPoly():
    y=datele_noastre["BILL_AMT2"].to_numpy().reshape(-1, 1)
    x=datele_noastre[["BILL_AMT1","BILL_AMT3"]].to_numpy().reshape(-1, 2)
    x1=datele_noastre["BILL_AMT1"].to_numpy().reshape(-1, 1)
    x2=datele_noastre["BILL_AMT3"].to_numpy().reshape(-1, 1)
    
    x_=PolynomialFeatures(degree=3, include_bias=True).fit_transform(x)
    #x_1=PolynomialFeatures(degree=2, include_bias=True).fit_transform(x1)
    #x_2=PolynomialFeatures(degree=2, include_bias=True).fit_transform(x2)
    model=LinearRegression().fit(x_, y)
    y_pred=model.predict(x_)
    
    r_sq=model.score(x_, y)
    print('coef R^2: ', r_sq)
    print('intercept: ', model.intercept_)
    print('slope: ', model.coef_)
    
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x1, x2, y, color='r')

    x1_range = np.linspace(x1.min(), x1.max(), 100)
    x2_range = np.linspace(x2.min(), x2.max(), 100)
    x1_mesh, x2_mesh = np.meshgrid(x1_range, x2_range)
    mesh_points = np.column_stack((x1_mesh.flatten(), x2_mesh.flatten()))
    y_pred_mesh = model.predict(PolynomialFeatures(degree=3, include_bias=True).fit_transform(mesh_points))
    y_pred_mesh = y_pred_mesh.reshape(x1_mesh.shape)
    
    ax.plot_surface(x1_mesh, x2_mesh, y_pred_mesh, alpha = 0.5)

    
    X_train, X_test, y_train, y_test = train_test_split(x_, y, test_size=0.3, random_state=0)
    regressor = LinearRegression()
    regressor.fit(X_train, y_train) # antrenarea algoritmului
    print(regressor.intercept_)
    print(regressor.coef_)
    #compararea valorilor reale cu cele prezise
    y_pred = regressor.predict(X_test)
    #Calcului Erorilor
    print('Eroarea medie absoluta:', metrics.mean_absolute_error(y_test, y_pred)) # MAE
    print('Eroare medie patratica:', metrics.mean_squared_error(y_test, y_pred)) # MSE 
    print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    
def regresiePoliMultivariata():   
    
    y=datele_noastre["BILL_AMT2"].to_numpy().reshape(-1, 1)
    x1=datele_noastre["BILL_AMT1"].to_numpy().reshape(-1, 1)
    x2=datele_noastre["BILL_AMT3"].to_numpy().reshape(-1, 1)
    x3=datele_noastre["BILL_AMT4"].to_numpy().reshape(-1, 1)
    
    x = np.concatenate((x1, x2, x3), axis=1)
    
    x_ = PolynomialFeatures(degree=3, include_bias=True).fit_transform(x)
    
    model = LinearRegression().fit(x_, y)
    
    r_sq = model.score(x_, y)
    intercept, coefficients = model.intercept_, model.coef_
    
    y_pred = model.predict(x_)
    print("coefficient of determinantion R^2: ", r_sq)
    print("intercept: ", intercept)
    print("coefficients: ", coefficients, sep="\n")
    
def regresieLiniaraMulti():   
    
    y=datele_noastre["BILL_AMT2"].to_numpy().reshape(-1, 1)
    x1=datele_noastre["BILL_AMT1"].to_numpy().reshape(-1, 1)
    x2=datele_noastre["BILL_AMT3"].to_numpy().reshape(-1, 1)
    x3=datele_noastre["BILL_AMT4"].to_numpy().reshape(-1, 1)
    
    x = np.concatenate((x1, x2, x3), axis=1)
    
    model = LinearRegression().fit(x, y)
    
    r_sq = model.score(x, y)
    intercept, coefficients = model.intercept_, model.coef_
    
    y_pred = model.predict(x)
    print("coefficient of determinantion R^2: ", r_sq)
    print("intercept: ", intercept)
    print("coefficients: ", coefficients, sep="\n")
    print()


def regresieLogistica():
    x = datele_noastre["PAY_0"].to_numpy().reshape(-1, 1)
    # y trebuie sa fie o coloana de 0 si de 1 in tabel
    y = datele_noastre["default payment next month"].to_numpy()

    model = LogisticRegression(solver='liblinear', random_state=0).fit(x,y)

    y_pred = model.predict(x)

    r_sq=model.score(x, y)
    print('coef R^2: ', r_sq)
    print('intercept:',model.intercept_)
    print('slope:',model.coef_)
    print('scor: ',model.score(x,y))
    print('Matrice de confuzie: \n',confusion_matrix(y, y_pred))
    print("Matthews_corrcoef: ", matthews_corrcoef(y,y_pred))
    print('Raport: \n',classification_report(y, y_pred))

def regresieLasso():
    x=datele_noastre["BILL_AMT1"].to_numpy().reshape(-1, 1)
    y=datele_noastre["BILL_AMT2"].to_numpy().reshape(-1, 1)
    
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    model = Lasso(alpha=0.3)
    model.fit(X_train/np.sqrt(20), y_train/np.sqrt(20))

    r_sq=model.score(x,y)
    intercept, coefficients = model.intercept_, model.coef_

    y_pred = model.predict(X_test)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='r')
    plt.plot(X_test, y_pred, color='b')
    plt.title(f'Regresie Lasso, R^2 = {r_sq}')
    plt.show()
    
    lasso = Lasso()
    lasso.fit(X_train, y_train) # antrenarea algoritmului
    print(lasso.intercept_)
    print(lasso.coef_)
    print('coef R^2:', r_sq)
    print('intercept:',intercept)
    print('slope:',coefficients, sep="\n")
    y_pred = lasso.predict(X_test)
    #compararea valorilor reale cu cele prezise
    mse = np.mean((y_pred-y_test))

    #Calcului Erorilor
    print('Eroarea medie absoluta:', metrics.mean_absolute_error(y_test, y_pred)) # MAE
    print('Eroare medie patratica:', metrics.mean_squared_error(y_test, y_pred)) # MSE print('RMSE:", np.sqrt(metrics.mean_squared_error(y_test, y_pred}}}
    print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print('MSE:', mse)
    
def regresieRidge():
    x=datele_noastre["BILL_AMT1"].to_numpy().reshape(-1, 1)
    y=datele_noastre["BILL_AMT2"].to_numpy().reshape(-1, 1)

    model = Ridge(alpha=0.05*np.sqrt(20))
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

    model.fit(X_train, y_train) # antrenarea algoritmului
    print("regresor intercept: ",model.intercept_)
    print("regresor slope: ",model.coef_)

    r_sq = model.score(x,y)
    print('coef R^2:', r_sq)
    print('intercept:',model.intercept_)
    print('slope:',model.coef_)
    #compararea valorilor reale cu cele prezise
    y_pred = model.predict(X_test)
    mse = np.mean((y_pred-y_test))

    #Calcului Erorilor
    print('Eroarea medie absoluta:', metrics.mean_absolute_error(y_test, y_pred)) # MAE
    print('Eroare medie patratica:', metrics.mean_squared_error(y_test, y_pred)) # MSE print('RMSE:", np.sqrt(metrics.mean_squared_error(y_test, y_pred}}}
    print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print('MSE:', mse)

    plt.scatter(x, y, color='red')
    plt.plot(X_test, y_pred)
    plt.title(f'Regresie Ridge, R^2 = {r_sq}')
    plt.show()
    
def regularizareElasticNet():
    x=datele_noastre["BILL_AMT2"].to_numpy().reshape(-1, 1)
    y=datele_noastre["BILL_AMT1"].to_numpy().reshape(-1, 1)

    model = ElasticNet(alpha=1.0 *np.sqrt(20), l1_ratio=0.5)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

    model.fit(X_train, y_train) # antrenarea algoritmului
    print("regresor intercept: ",model.intercept_)
    print("regresor slope: ",model.coef_)

    r_sq = model.score(x,y)
    print('coef R^2:', r_sq)
    print('intercept:',model.intercept_)
    print('slope:',model.coef_)
    #compararea valorilor reale cu cele prezise
    y_pred = model.predict(X_test)
    mse = np.mean(y_pred-y_test)

    #Calcului Erorilor
    print('Eroarea medie absoluta:', metrics.mean_absolute_error(y_test, y_pred)) # MAE
    print('Eroare medie patratica:', metrics.mean_squared_error(y_test, y_pred)) # MSE print('RMSE:", np.sqrt(metrics.mean_squared_error(y_test, y_pred}}}
    print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print('MSE:', mse)

    plt.title(f'ElasticNet, R^2 = {r_sq}')
    plt.scatter(x, y, color='red')
    plt.plot(X_test, y_pred)
    plt.show()


print(curatareDate())
print(datele_noastre)
print()

print(vizualizareDate())

print('Regresie Liniara Simpla: ')
print()
print(regresieLiniaraS())
print()
print('Regresie Univariata Polinomiala: ')
print()
print(regresiePoliUni())
print()
print('Regresia Bivariata Simpla: ')
print()
print(regresieBivariataS())
print()
print('Regresie Bivaiata Polinomiala:')
print()
print(regresieBivariataPoly())
print()
print('Regresie Multivariata Polinomiala:')
print()
print(regresiePoliMultivariata())
print()
print()
print('Regresie Liniara Multivariata:')
print()
print(regresieLiniaraMulti())
print()
print('Regresie Logistica')
print(regresieLogistica())
print()
print('Regresie Lasso')
print(regresieLasso())
print()
print('Regresie Ridge:')
print(regresieRidge())
print()
print('Regularizare ElasticNet:')
print(regularizareElasticNet())
print('\n')


