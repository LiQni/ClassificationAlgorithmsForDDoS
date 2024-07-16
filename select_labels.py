import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression

data=pd.read_csv(r'ddos_dataset.csv')
X = data.drop(['Label'], axis=1)  # 特征
y = data['Label']  # 标签
print("load data ok")

k = 25
selector = SelectKBest(score_func=f_regression, k=k)
X_new = selector.fit_transform(X, y)
print("X_new ok")
mask = selector.get_support()  # 获得特征掩码
print("mask ok")
new_features = X.columns[mask]  # 选择重要的特征
print("new_features ok")
print(new_features)