from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression  #  LogisticRegression 면접질문에서 많이나옴 (분류모델이라고 외우면됨)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


import numpy as np
from sklearn import datasets
from sklearn.datasets import load_wine
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split, KFold, cross_val_score
datasets = load_wine()
# print(datasets.DESCR)
# print(datasets.feature_names)

x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

n_splits=5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

model = LinearSVC()
# acc:  [0.72413793 0.96551724 0.75       0.89285714 0.89285714] 평균 :  0.8451
# model = SVC()
# acc:  [0.5862069  0.65517241 0.5        0.67857143 0.67857143] 평균 :  0.6197
# model = KNeighborsClassifier()
# acc:  [0.65517241 0.79310345 0.57142857 0.71428571 0.5       ] 평균 :  0.6468
# model = LogisticRegression()
# acc:  [0.89655172 1.         0.82142857 0.89285714 0.96428571] 평균 :  0.915
# model = DecisionTreeClassifier()
# acc:  [0.79310345 0.75862069 0.85714286 0.85714286 0.85714286] 평균 :  0.8246
# model = DecisionTreeRegressor()
# acc:  [0.4486692  0.6340694  0.888      0.44       0.86634845] 평균 :  0.6554
# model = RandomForestClassifier()
# acc:  [0.96551724 1.         0.96428571 0.92857143 0.96428571] 평균 :  0.9645
# model = RandomForestRegressor()
# acc:  [0.72283498 0.85163344 0.8681984  0.92720933 0.88346253] 평균 :  0.8507


scores = cross_val_score(model, x_train, y_train, cv=kfold)
print("acc: " ,scores, "평균 : ",round(np.mean(scores), 4))