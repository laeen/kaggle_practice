import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn import linear_model, svm, gaussian_process
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

from sklearn.externals import joblib

data_train = pd.read_csv("cleaning.csv")

y = data_train['SalePrice'].copy().values
x = data_train

del x['SalePrice']

y_Scaler = preprocessing.StandardScaler()

x_scaled = preprocessing.StandardScaler().fit_transform(x)
y_scaled = y_Scaler.fit_transform(y.reshape(-1, 1))

joblib.dump(y_Scaler, 'scalarY')


X_train, X_test, y_train, y_test = train_test_split(
    x_scaled,
    y_scaled,
    test_size=0.33,
    random_state=42)

clfs = {
    'svm': svm.SVR(),
    'RandomForestRegressor': RandomForestRegressor(n_estimators=400),
    'BayesianRidge': linear_model.BayesianRidge()
}
for clf in clfs:
    try:
        clfs[clf].fit(X_train, y_train)
        y_pred = clfs[clf].predict(X_test)
        print(clf + " cost:" + str(np.sum(y_pred - y_test) / len(y_pred)))
    except Exception as e:
        print(clf + " Error:")
        print(str(e))

joblib.dump(clfs['RandomForestRegressor'], "train_model.m")