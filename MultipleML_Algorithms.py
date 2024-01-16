# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 20:12:41 2023

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
from sklearn.decomposition import PCA
from sklearn.utils import resample
from sklearn.manifold import TSNE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LinearRegression
import time
from sklearn.naive_bayes import GaussianNB

datele_noastre = pd.read_csv('C:\\Users\\eu\\Desktop\\FMI\\MVMOD\\InvatareAutomata\\smoke_detection_iot.csv')


def vizualizareDate():
    
    datele_noastre.corr()
    plt.figure(dpi=100)
    plt.subplots(figsize=(10,7))
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
    print()
    datele_noastre.drop('Unnamed: 0', inplace=True, axis=1)
    
# def Linear_SVR():
#     x=datele_noastre["PM2.5"].to_numpy().reshape(-1, 1)
#     y=datele_noastre["NC2.5"].to_numpy().reshape(-1, 1)

#     svm_reg = LinearSVR(epsilon=1.5).fit(x, y) 
#     y_pred = svm_reg.predict(x)
#     #X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
#     r_sq = svm_reg.score(x, y)
#     mse = mean_squared_error(y, y_pred)

#     print('R^2:', r_sq)
#     print('MSE:', mse)
#     print('Intercept:', svm_reg.intercept_)
#     print('Slope:', svm_reg.coef_)

#     plt.figure(figsize=(10, 6))
#     plt.scatter(x, y, color='r')
#     plt.plot(x, y_pred, color='b')
#     plt.title(f'SVR liniar: R^2 = {r_sq}')
#     plt.show()
# def regresieLiniaraS():
#     x=datele_noastre["NC2.5"].to_numpy().reshape(-1, 1)
#     y=datele_noastre["PM2.5"].to_numpy().reshape(-1, 1)
    
#     model=LinearRegression().fit(x, y)
#     y_pred=model.predict(x)
    
#     r_sq=model.score(x, y)
#     print('coef R^2: ', r_sq)
#     print('intercept: ', model.intercept_)
#     print('slope: ', model.coef_)
    
#     X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
#     regressor = LinearRegression()
#     regressor.fit(X_train, y_train) # antrenarea algoritmului
#     print(regressor.intercept_)
#     print(regressor.coef_)
#     #compararea valorilor reale cu cele prezise
#     y_pred = regressor.predict(X_test)
#     #Calcului Erorilor
#     print('Eroarea medie absoluta:', metrics.mean_absolute_error(y_test, y_pred)) # MAE
#     print('Eroare medie patratica:', metrics.mean_squared_error(y_test, y_pred)) # MSE 
#     print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    
#     plt.figure(figsize=(10, 6))
#     plt.scatter(x, y, color='red')
#     plt.plot(X_test, y_pred, color='blue')
#     plt.title(f'Regresie Liniara Simpla: R^2 = {r_sq}')
#     plt.show()

def regresieLiniaraS():
    start_time = time.time()  # Începutul măsurătorii timpului

    x = datele_noastre["PM2.5"].to_numpy().reshape(-1, 1)
    y = datele_noastre["PM1.0"].to_numpy().reshape(-1, 1)
    
    # Aplică o transformare logaritmică asupra variabilelor
    x_transformed = np.log1p(x)
    y_transformed = np.log1p(y)
    
    from sklearn.linear_model import LinearRegression
    model = LinearRegression().fit(x_transformed, y_transformed)
    y_pred_transformed = model.predict(x_transformed)
    
    r_sq = model.score(x_transformed, y_transformed)
    print('coef R^2: ', r_sq)
    print('intercept: ', model.intercept_)
    print('slope: ', model.coef_)
    
    X_train, X_test, y_train, y_test = train_test_split(x_transformed, y_transformed, test_size=0.3, random_state=0)
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    
    print('Intercept:', regressor.intercept_)
    print('Slope:', regressor.coef_)
    
    # Compararea valorilor reale cu cele prezise
    y_pred = regressor.predict(X_test)
    
    # Calculul erorilor
    print('Eroarea medie absoluta:', metrics.mean_absolute_error(y_test, y_pred))  # MAE
    print('Eroare medie patratica:', metrics.mean_squared_error(y_test, y_pred))  # MSE 
    print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    
    # Plotarea rezultatelor
    plt.figure(figsize=(10, 6))
    plt.scatter(x_transformed, y_transformed, color='red', label='Date reale')
    plt.plot(X_test, y_pred, color='blue', label='Regresie liniară')
    plt.title(f'Regresie Liniara Simpla: R^2 = {r_sq}')
    plt.xlabel('Log(PM2.5 + 1)')
    plt.ylabel('Log(PM1.0 + 1)')
    plt.legend()
    plt.show()
    end_time = time.time()  # Sfârșitul măsurătorii timpului
    elapsed_time = end_time - start_time
    print(f'Timpul de procesare: {elapsed_time} secunde')
    
# def regresiePoliUni():
#     x=datele_noastre["NC2.5"].to_numpy().reshape(-1, 1)
#     y=datele_noastre["PM2.5"].to_numpy()

#     x_= PolynomialFeatures(degree=3, include_bias=True).fit_transform(x)
#     model= LinearRegression().fit(x_, y)

#     r_sq=model.score(x_,y)
#     intercept, coefficients = model.intercept_, model.coef_
    
#     x_range = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
#     x_range_poly= PolynomialFeatures(degree=3, include_bias=True).fit_transform(x_range)
#     y_pred = model.predict(x_range_poly)
#     mse = mean_squared_error(y, model.predict(x_))
    
#     plt.figure(figsize=(10, 6))
#     plt.scatter(x, y, color='r')
#     plt.plot(x_range, y_pred, color='b')
#     plt.title(f'Regresie Univariata Polinomiala: R^2 = {r_sq}')
#     plt.show()
    
#     X_train, X_test, y_train, y_test = train_test_split(x_, y, test_size=0.3, random_state=0)
#     regressor = LinearRegression()
#     regressor.fit(X_train, y_train) # antrenarea algoritmului
#     #print(regressor.intercept_)
#     #print(regressor.coef_)
#     print('coef R^2:', r_sq)
#     print('MSE:', mse)
#     print('intercept:',intercept)
#     print('slope:',coefficients, sep="\n")
#     y_pred = regressor.predict(X_test)
def regresiePoliUni():
    start_time = time.time()  # Începutul măsurătorii timpului

    x = datele_noastre["PM2.5"].to_numpy().reshape(-1, 1)
    y = datele_noastre["PM1.0"].to_numpy()

    # Aplică o transformare logaritmică asupra ambelor variabile
    x_transformed = np.log1p(x)
    y_transformed = np.log1p(y)

    # Transformarea polinomială pentru variabila independentă
    x_poly = PolynomialFeatures(degree=3, include_bias=True).fit_transform(x_transformed)

    # Regresie liniară pe datele transformate
    model = LinearRegression().fit(x_poly, y_transformed)

    r_sq = model.score(x_poly, y_transformed)
    intercept, coefficients = model.intercept_, model.coef_

    # Crearea unui set de date pentru a realiza predicții și plota graficul
    x_range = np.linspace(x_transformed.min(), x_transformed.max(), 100).reshape(-1, 1)
    x_range_poly = PolynomialFeatures(degree=3, include_bias=True).fit_transform(x_range)
    y_pred_transformed = model.predict(x_range_poly)

    mse = mean_squared_error(y_transformed, model.predict(x_poly))

    # Plotarea rezultatelor
    plt.figure(figsize=(10, 6))
    plt.scatter(x_transformed, y_transformed, color='r', label='Date reale')
    plt.plot(x_range, y_pred_transformed, color='b', label='Regresie polinomială')
    plt.title(f'Regresie Univariata Polinomiala: R^2 = {r_sq}')
    plt.xlabel('Log(PM2.5 + 1)')
    plt.ylabel('Log(PM1.0 + 1)')
    plt.legend()
    plt.show()

    # Divizarea setului de date și evaluarea performanței
    X_train, X_test, y_train, y_test = train_test_split(x_poly, y_transformed, test_size=0.3, random_state=0)
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    print('coef R^2:', r_sq)
    print('MSE:', mse)
    print('intercept:', intercept)
    print('slope:', coefficients, sep="\n")

    y_pred_transformed = regressor.predict(X_test)

    # Revino la scala originală pentru a evalua erorile pe scala originală
    y_pred = np.expm1(y_pred_transformed)
    y_test_original = np.expm1(y_test)
    mse_original = mean_squared_error(y_test_original, y_pred)

    print('MSE pe scala originală:', mse_original)
    
    end_time = time.time()  # Sfârșitul măsurătorii timpului
    elapsed_time = end_time - start_time
    print(f'Timpul de procesare: {elapsed_time} secunde')

    # Restul codului pentru evaluarea performanței pe setul de test și altele

def regresieBivariataS():
    start_time = time.time()  # Începutul măsurătorii timpului
    
    x1 = datele_noastre["PM2.5"].to_numpy().reshape(-1, 1)
    x2 = datele_noastre["NC1.0"].to_numpy().reshape(-1, 1)
    y = datele_noastre["PM1.0"].to_numpy()

    # Aplică o transformare logaritmică asupra tuturor variabilelor
    x1_transformed = np.log1p(x1)
    x2_transformed = np.log1p(x2)
    y_transformed = np.log1p(y)

    x = np.concatenate((x1_transformed, x2_transformed), axis=1)

    model = LinearRegression().fit(x, y_transformed)
    y_pred_transformed = model.predict(x)

    r_sq = model.score(x, y_transformed)
    
    # Calcularea erorilor pe datele transformate
    mae = metrics.mean_absolute_error(y_transformed, y_pred_transformed)
    mse = metrics.mean_squared_error(y_transformed, y_pred_transformed)
    rmse = np.sqrt(mse)

    print('Coefficient R^2:', r_sq)
    print('Intercept:', model.intercept_)
    print('Slope:', model.coef_)
    print('Mean Absolute Error:', mae)
    print('Mean Squared Error:', mse)
    print('Root Mean Squared Error:', rmse)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x1_transformed, x2_transformed, y_transformed, color='r')

    x1_range = np.linspace(x1_transformed.min(), x1_transformed.max(), 100)
    x2_range = np.linspace(x2_transformed.min(), x2_transformed.max(), 100)
    x1_mesh, x2_mesh = np.meshgrid(x1_range, x2_range)
    x_mesh = np.concatenate((x1_mesh.flatten().reshape(-1, 1), x2_mesh.flatten().reshape(-1, 1)), axis=1)
    y_pred_mesh_transformed = model.predict(x_mesh)
    y_pred_mesh_transformed = y_pred_mesh_transformed.reshape(x1_mesh.shape)

    ax.plot_surface(x1_mesh, x2_mesh, y_pred_mesh_transformed, alpha=0.5)

    end_time = time.time()  # Sfârșitul măsurătorii timpului
    elapsed_time = end_time - start_time
    print(f'Timpul de procesare pentru regresia bivariată: {elapsed_time} secunde')

    plt.show()




# def Decision_Tree_Regression():
#     start_time = time.time()  # Începutul măsurătorii timpului

#     x=datele_noastre["NC0.5"].to_numpy().reshape(-1, 1)
#     y=datele_noastre["PM1.0"].to_numpy()
    
#     regressor = Pipeline ([
#         ("scaler", StandardScaler()),
#         ("regressor", DecisionTreeRegressor(random_state=0))
#     ]).fit(x, y) 
    
#     y_pred = regressor.predict(x)
    
#     r_sq = regressor.score(x, y)
#     mse = mean_squared_error(y, y_pred)
   
#     print('R^2:', r_sq)
#     print('RMSE:', np.sqrt(metrics.mean_squared_error(y, y_pred)))
#     print('MSE:', mse)
    
#     tree.plot_tree(regressor["regressor"])
#     plt.savefig('Regressor_tree.pdf')
#     plt.show()
#     end_time = time.time()  # Sfârșitul măsurătorii timpului
#     elapsed_time = end_time - start_time
#     print(f'Timpul de procesare: {elapsed_time} secunde')



def Random_Forest_Regression():
    start_time = time.time()  # Începutul măsurătorii timpului

    x = datele_noastre["PM2.5"].to_numpy().reshape(-1, 1)
    y = datele_noastre["PM1.0"].to_numpy()

    # Aplică o transformare logaritmică asupra ambelor variabile
    x_transformed = np.log1p(x)
    y_transformed = np.log1p(y)

    regressor = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", RandomForestRegressor(max_depth=2, random_state=0))
    ]).fit(x_transformed, y_transformed)

    y_pred_transformed = regressor.predict(x_transformed)

    # Calcularea R^2 pe datele transformate
    r_sq = regressor.score(x_transformed, y_transformed)
    
    # Calcularea MSE pe datele transformate
    mse = mean_squared_error(y_transformed, y_pred_transformed)

    print('R^2:', r_sq)
    print('RMSE:', np.sqrt(metrics.mean_squared_error(y_transformed, y_pred_transformed)))
    print('MSE:', mse)

    end_time = time.time()  # Sfârșitul măsurătorii timpului
    elapsed_time = end_time - start_time
    print(f'Timpul de procesare pentru Random Forest Regression: {elapsed_time} secunde')


def Linear_SVR():
    
    start_time = time.time()  # Începutul măsurătorii timpului

    x = datele_noastre["PM2.5"].to_numpy().reshape(-1, 1)
    y = datele_noastre["PM1.0"].to_numpy()

    # Aplică o transformare logaritmică asupra ambelor variabile
    x_transformed = np.log1p(x)
    y_transformed = np.log1p(y)
    
    svm_reg = LinearSVR(epsilon=1.5).fit(x_transformed, y_transformed) 
    y_pred_transformed = svm_reg.predict(x_transformed)
    
    # Calcularea R^2 pe datele transformate
    r_sq = svm_reg.score(x_transformed, y_transformed)
    
    # Calcularea MSE pe datele transformate
    mse = mean_squared_error(y_transformed, y_pred_transformed)

    print('R^2:', r_sq)
    print('MSE:', mse)
    print('RMSE:', np.sqrt(metrics.mean_squared_error(y_transformed, y_pred_transformed)))
    print('Intercept:', svm_reg.intercept_)
    print('Slope:', svm_reg.coef_)

    # Plotarea rezultatelor pe datele transformate
    plt.figure(figsize=(10, 6))
    plt.scatter(x_transformed, y_transformed, color='r')
    plt.plot(x_transformed, y_pred_transformed, color='b')
    plt.title(f'SVR liniar: R^2 = {r_sq}')
    plt.xlabel('Log(PM2.5 + 1)')
    plt.ylabel('Log(PM1.0 + 1)')
    plt.show()

    end_time = time.time()  # Sfârșitul măsurătorii timpului
    elapsed_time = end_time - start_time
    print(f'Timpul de procesare pentru SVR liniar: {elapsed_time} secunde')
    
#----------------------------------------------------------------------------------------------------------

def Random_Forest_Classification():
    start_time = time.time()  # Începutul măsurătorii timpului
    
    # Selectarea variabilelor de intrare și de ieșire
    x_cols = ["Humidity[%]", "Raw H2", "PM2.5", "NC2.5"]
    y_col = "Fire Alarm"

    x = datele_noastre[x_cols].to_numpy().reshape(-1, len(x_cols))
    y = datele_noastre[y_col].to_numpy()

    # Aplică o transformare logaritmică asupra variabilelor de intrare
    x_transformed = np.log1p(x)

    # Antrenarea modelului Random Forest
    X_train, X_test, y_train, y_test = train_test_split(x_transformed, y, test_size=0.3, random_state=0)
    
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=100))
    ]).fit(X_train, y_train)

    # Realizează predicții
    y_pred = clf.predict(X_test)

    # Calcularea erorii medii
    mse = np.mean(y_pred - y_test)

    # Evaluarea performanței modelului
    print("Matthews correlation coefficient:", matthews_corrcoef(y_test, y_pred))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print()
    print("Classification report:")
    print(classification_report(y_test, y_pred))
    print()

    # Afișarea timpului de procesare
    end_time = time.time()  # Sfârșitul măsurătorii timpului
    elapsed_time = end_time - start_time
    print(f'Timpul de procesare: {elapsed_time} secunde')

def Decision_Tree_Classification():
    start_time = time.time()  # Începutul măsurătorii timpului

    x_cols = ["Humidity[%]", "Raw H2", "PM2.5", "NC2.5"]
    y_col = "Fire Alarm"

    x = datele_noastre[x_cols].to_numpy().reshape(-1, len(x_cols))
    y = datele_noastre[y_col].to_numpy()

    # Aplică o transformare logaritmică asupra variabilelor continue
    x_transformed = np.log1p(x)

    # Aplică PolynomialFeatures după transformare
    x_transformed_poly = PolynomialFeatures(degree=3, include_bias=True).fit_transform(x_transformed)

    X_train, X_test, y_train, y_test = train_test_split(x_transformed_poly, y, test_size=0.3, random_state=0)
    
    clf_gini = Pipeline([
        ("scaler", StandardScaler()),
        ("clf_gini", DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=3, min_samples_leaf=5))
    ]).fit(X_train, y_train) 

    y_pred = clf_gini.predict(X_test)

    print("matthews_corrcoef: ", matthews_corrcoef(y_test, y_pred))
    print("Confusion matrix: ")
    print(confusion_matrix(y_test, y_pred))
    print()
    print("Classification report: ")
    print(classification_report(y_test, y_pred))
    print()

    tree.plot_tree(clf_gini["clf_gini"])
    plt.savefig('Class_tree.pdf')
    plt.show()

    end_time = time.time()  # Sfârșitul măsurătorii timpului
    elapsed_time = end_time - start_time
    print(f'Timpul de procesare: {elapsed_time} secunde')

def svc():
    start_time = time.time()  # Începutul măsurătorii timpului

    x_cols = ["Humidity[%]", "Raw H2", "PM2.5", "NC2.5"]
    y_col = "Fire Alarm"

    x = datele_noastre[x_cols].to_numpy().reshape(-1, len(x_cols))
    y = datele_noastre[y_col].to_numpy()

    # Aplică o transformare logaritmică asupra variabilelor continue
    x_transformed = np.log1p(x)

    # Aplică PolynomialFeatures după transformare
    x_transformed_poly = PolynomialFeatures(degree=2, include_bias=True).fit_transform(x_transformed)

    svc_clas = Pipeline([
        ("scaler", StandardScaler()),
        ("svc_clas", SVC(kernel="poly", degree=2, C=100, coef0=1))
    ]).fit(x_transformed_poly, y)

    y_pred = svc_clas.predict(x_transformed_poly)

    print("matthews_corrcoef: ", matthews_corrcoef(y, y_pred))
    print()
    print("Confusion matrix: ")
    print(confusion_matrix(y, y_pred))
    print()
    print("Classification report: ")
    print(classification_report(y, y_pred))
    print()

    end_time = time.time()  # Sfârșitul măsurătorii timpului
    elapsed_time = end_time - start_time
    print(f'Timpul de procesare: {elapsed_time} secunde')
    
def Gaussian():
    start_time = time.time()  # Începutul măsurătorii timpului

    x_cols = ["Humidity[%]", "Raw H2", "PM2.5", "NC2.5"]
    y_col = "Fire Alarm"

    x = datele_noastre[x_cols].to_numpy().reshape(-1, len(x_cols))
    y = datele_noastre[y_col].to_numpy()

    # Aplică o transformare logaritmică asupra variabilelor continue
    x_transformed = np.log1p(x)

    # Aplică PolynomialFeatures după transformare
    x_transformed_poly = PolynomialFeatures(degree=3, include_bias=True).fit_transform(x_transformed)

    X_train, X_test, y_train, y_test = train_test_split(x_transformed_poly, y, test_size=0.3, random_state=0)

    GAUSS = Pipeline([
        ("scaler", StandardScaler()),
        ("GAUSS", GaussianNB())
    ]).fit(X_train, y_train)

    y_pred = GAUSS.predict(X_test)

    r_sq = GAUSS.score(x_transformed_poly, y)

    print("matthews_corrcoef: ", matthews_corrcoef(y_test, y_pred))
    print("Confusion matrix: ")
    print(confusion_matrix(y_test, y_pred))
    print()
    print("Classification report: ")
    print(classification_report(y_test, y_pred))
    print()

    end_time = time.time()  # Sfârșitul măsurătorii timpului
    elapsed_time = end_time - start_time
    print(f'Timpul de procesare: {elapsed_time} secunde')





print(curatareDate())    
print(vizualizareDate())

# print('Regresie Liniara Simpla: ')
# print(regresieLiniaraS())
# print('\n')

# print('Regresie Univariata Polinomiala: ')
# print(regresiePoliUni())
# print('\n')

# print('Regresie Liniara Bivariata:')
# print(regresieBivariataS())
# print('\n')

# print('Decision tree regression: ')
# print(Decision_Tree_Regression())
# print('\n')

# print('SVR liniar: ')
# print(Linear_SVR())
# print('\n')

# print('Random forest regression: ')
# print(Random_Forest_Regression())
# print('\n')

print('Random forest classification: ')
print(Random_Forest_Classification())
print('\n')

print('Decision tree classification: ')
print(Decision_Tree_Classification())
print('\n')

# print('SVC: ')
# print(svc())
# print('\n')

print('Gaussian Naive Bayes: ')
print(Gaussian())
print('\n')



