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
    
    
    
def Decision_Tree_Classification():
    x=datele_noastre[["BILL_AMT1","BILL_AMT2","BILL_AMT3","BILL_AMT4"]].to_numpy().reshape(-1, 4)
    y=datele_noastre["default payment next month"].to_numpy()
    
    
    x = PolynomialFeatures(degree=3, include_bias=True).fit_transform(x)
    
    
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    clf_gini = Pipeline ([
        ("scaler", StandardScaler()),
        ("clf_gini", DecisionTreeClassifier(criterion="gini", random_state = 100, max_depth = 3, min_samples_leaf=5))
    ]).fit(X_train, y_train) 
    
    
    y_pred = clf_gini.predict(X_test)
    
    r_sq = clf_gini.score(x, y)
    
    print('R^2:', r_sq)
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

def Decision_Tree_Regression():
    x=datele_noastre["PAY_4"].to_numpy().reshape(-1, 1)
    y=datele_noastre["PAY_5"].to_numpy()
    
    # x=datele_noastre["BILL_AMT2"].to_numpy().reshape(-1, 1)
    # y=datele_noastre["BILL_AMT1"].to_numpy()
    
    # x = PolynomialFeatures(degree=2, include_bias=True).fit_transform(x)
    
    #X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    
    regressor = Pipeline ([
        ("scaler", StandardScaler()),
        ("regressor", DecisionTreeRegressor(random_state=0))
    ]).fit(x, y) 
    
    y_pred = regressor.predict(x)
    
    r_sq = regressor.score(x, y)
    mse = mean_squared_error(y, y_pred)
   
    print('R^2:', r_sq)
    print('RMSE:', np.sqrt(metrics.mean_squared_error(y, y_pred)))
    print('MSE:', mse)
    
    tree.plot_tree(regressor["regressor"])
    plt.savefig('Regressor_tree.pdf')
    plt.show()

    
def Random_Forest_Classification():
    x=datele_noastre[["BILL_AMT1","BILL_AMT2","BILL_AMT3","BILL_AMT4"]].to_numpy().reshape(-1, 4)
    y=datele_noastre["default payment next month"].to_numpy()
    
    x = PolynomialFeatures(degree=3, include_bias=True).fit_transform(x)
    
    
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    clf = Pipeline ([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators = 100))
    ]).fit(X_train, y_train) 

    
    y_pred = clf.predict(X_test)
    
    r_sq = clf.score(x, y)
    mse = np.mean(y_pred-y_test)
    
    print('R^2:', r_sq)
    print("matthews_corrcoef: ", matthews_corrcoef(y_test, y_pred))
    print("Confusion matrix: ")
    print(confusion_matrix(y_test, y_pred))
    print()
    print("Classification report: ")
    print(classification_report(y_test, y_pred))
    print()

    
def Random_Forest_Regression():
    x=datele_noastre["PAY_4"].to_numpy().reshape(-1, 1)
    y=datele_noastre["PAY_5"].to_numpy()
    
    # x = PolynomialFeatures(degree=2, include_bias=True).fit_transform(x)
    
    
    regressor = Pipeline ([
        ("scaler", StandardScaler()),
        ("regressor", RandomForestRegressor(max_depth=2, random_state=0))
    ]).fit(x, y) 
    
    y_pred = regressor.predict(x)
    
    r_sq = regressor.score(x, y)
    mse = mean_squared_error(y, y_pred)
    
    print('R^2:', r_sq)
    print('RMSE:', np.sqrt(metrics.mean_squared_error(y, y_pred)))
    print('MSE:', mse)


def Kfold():
    x=datele_noastre["PAY_6"].to_numpy().reshape(-1, 1)
    y=datele_noastre["default payment next month"].to_numpy()
    
    # x = PolynomialFeatures(degree=2, include_bias=True).fit_transform(x)
    
    clf = DecisionTreeClassifier(random_state=100)
    model = KFold(n_splits=5, shuffle=False)
    
    scores = cross_val_score(clf, x, y, cv = model)
    
    print('Scores:', scores)  
    print("Average CV Score: ", scores.mean())
    print("Nr splits: ", len(scores))
  

print(curatareDate())    
print(vizualizareDate())

print('Decision tree classification: ')
print(Decision_Tree_Classification())
print('\n')

print('Decision tree regression: ')
print(Decision_Tree_Regression())
print('\n')

print('Random forest classification: ')
print(Random_Forest_Classification())
print('\n')

print('Random forest regression: ')
print(Random_Forest_Regression())
print('\n')

print('Kfold: ')
print(Kfold())
print('\n')