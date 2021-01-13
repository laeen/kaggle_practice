import pickle
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from scipy import stats
from scipy.stats import norm
from sklearn import preprocessing

import warnings

warnings.filterwarnings('ignore')

print("this is cleaning work")
data_train = pd.read_csv("data/train.csv")
data_test = pd.read_csv("data/test.csv")

# data_train['SalePrice'].describe()
# # count      1460.000000
# # mean     180921.195890
# # std       79442.502883
# # min       34900.000000
# # 25%      129975.000000
# # 50%      163000.000000
# # 75%      214000.000000
# # max      755000.000000
#
#
# data_train['SalePrice'].skew()
# # 6.536281860064529
# data_train['SalePrice'].kurt()
# # 1.8828757597682129


# corrmat = data_train.corr()
# # f, ax = plt.subplots(figsize=(20, 9))
# # sns.heatmap(corrmat, vmax=0.8, square=True)
#
# k  = 20 # 关系矩阵中将显示20个特征
# cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
# cm = np.corrcoef(data_train[cols].values.T)
# sns.set(font_scale=1)
# hm = sns.heatmap(cm,
#                  cbar=True,
#                  annot=True,
#                  square=True,
#                  fmt='.2f',
#                  annot_kws={'size': 6},
#                  yticklabels=cols.values,
#                  xticklabels=cols.values
#                  )
# plt.show()

# #
# # temp_key = 'Alley'
# # temp = data_train['Alley'].value_counts()
#
# temp_key = 'MSZoning'

#
df = data_train[
    ['OverallQual', 'GrLivArea', 'TotalBsmtSF', 'GarageArea', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'YearRemodAdd',
     'SalePrice']]

df_test = data_test[
    ['OverallQual', 'GrLivArea', 'TotalBsmtSF', 'GarageArea', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'YearRemodAdd']]

f_names = ['YearBuilt', 'CentralAir', 'Neighborhood', 'RoofMatl', 'HouseStyle', 'KitchenQual', 'SaleCondition',
           'SaleType']


# # temp_key = 'Alley'
# # temp = data_train['Alley'].value_counts()
data_test['KitchenQual'].value_counts()
for x in f_names:
    label = preprocessing.LabelEncoder()

    print("x= ", x, "\ttypeA: ", type(data_train[x]), "\ttypeB: ",type(data_test[x]))
    # temp_df = pd.merge(data_train[x], data_test[x])
    # label.fit_transform(temp_df)



    data_train[x].fillna(data_train[x].mode().iloc[0])
    data_test[x].fillna(data_test[x].mode().iloc[0])

    df[x] = label.fit_transform(data_train[x])
    df_test[x] = label.fit_transform(data_test[x])


# try:
#     data_train[x].fillna(data_train[x].mode())
#     d = label.fit_transform(data_train[x])
# except Exception as e:
#     print("x=", x, "\t", e)

# 归一化

df.to_csv('./cleaning.csv', index=False)
df_test.to_csv('./cleaning_test.csv', index=False)


