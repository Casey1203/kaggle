# coding: utf-8

import pandas as pd
from mxnet import nd

train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')

all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))

# print(all_features.head())

numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
# print('numeric_features_index', numeric_features)
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))
# 标准化后，每个特征的均值变为0，所以可以直接用0来替换缺失值
all_features[numeric_features] = all_features[numeric_features].fillna(0)

all_features = pd.get_dummies(all_features, dummy_na=True) # 会生成“列名_nan”

n_train = train_data.shape[0]
train_features = nd.array(all_features[:n_train].values)
test_features = nd.array(all_features[n_train:].values)
train_labels = nd.array(train_data.SalePrice.values).reshape((-1, 1))


