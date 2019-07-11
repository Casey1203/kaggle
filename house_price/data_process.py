# coding: utf-8

import pandas as pd
from mxnet import nd

train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')
remove_feature_list = [
    'BldgType_nan',
    'CentralAir_nan',
    'Condition1_nan',
    'Condition2_nan',
    'ExterCond_nan',
    'Exterior1st_nan',
    'Exterior2nd_nan',
    'ExterQual_nan',
    'Functional_nan',
    'HeatingQC_nan',
    'HouseStyle_nan',
    'KitchenQual_nan',
    'LandContour_nan',
    'LandSlope_nan',
    'MSZoning_nan',
    'LotConfig_nan',
    'Neighborhood_nan',
    'LotShape_nan',
    'PavedDrive_nan',
    'RoofMatl_nan',
    'RoofStyle_nan',
    'SaleCondition_nan',
    'SaleType_nan',
    'Street_nan',
    'Utilities_nan',
    'Foundation_nan'
]

all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))

# print(all_features.head())

numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
# print('numeric_features_index', numeric_features)
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))
# 标准化后，每个特征的均值变为0，所以可以直接用0来替换缺失值
all_features[numeric_features] = all_features[numeric_features].fillna(0)
all_features = pd.get_dummies(all_features, dummy_na=True) # 会生成“列名_nan”
print(all_features.shape)
feature_list = list(set(all_features.columns.tolist()) - set(remove_feature_list))
all_features = all_features[feature_list]
# all_features = all_features.apply(
#     lambda x: (x - x.mean()) / (x.std()))
# all_features = all_features.subtract(all_features.mean())
print(all_features.shape)

n_train = train_data.shape[0]
train_features = nd.array(all_features[:n_train].values)
test_features = nd.array(all_features[n_train:].values)
train_labels = nd.array(train_data.SalePrice.values).reshape((-1, 1))

feature_list = all_features.columns.tolist()


