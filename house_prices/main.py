import pickle
import pandas as pd

from sklearn.externals import joblib
from sklearn import preprocessing

clf = joblib.load("train_model.m")

data_test = pd.read_csv("cleaning_test.csv")
df = pd.read_csv("data/test.csv")

f_names = ['OverallQual', 'GrLivArea', 'TotalBsmtSF', 'GarageArea', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd',
           'YearRemodAdd', 'YearBuilt', 'CentralAir', 'Neighborhood', 'RoofMatl', 'HouseStyle', 'KitchenQual',
           'SaleCondition', 'SaleType']

for key in f_names:
    data_test[key].fillna(data_test[key].mode()[0], inplace=True)


# 读取模型参数，对测试进行再编码

x = data_test.values
y_te_pred = clf.predict(x)

y_scaler = joblib.load('scalarY')

prediction = pd.DataFrame(y_te_pred, columns=['SalePrice'])

p = y_scaler.inverse_transform(prediction)

result = pd.concat([df['Id'], pd.DataFrame(p, columns=['SalePrice'])], axis=1)
print(type(p), type(result), type(prediction))

result.to_csv('./Predictions.csv', index=False)
