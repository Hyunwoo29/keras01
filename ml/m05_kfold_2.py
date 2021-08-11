import numpy as np
from sklearn import datasets
from sklearn.datasets import load_iris
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression #  LogisticRegression 면접질문에서 많이나옴 (분류모델이라고 외우면됨)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

datasets = load_iris()

x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

n_splits=5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)


# model = LinearSVC()
# 1. [0.96666667 0.96666667 1.         0.9        1.        ]
# 2. acc:  [1.         0.95833333 0.95833333 1.         0.91666667] 평균 :  0.9667
# model = SVC()
# 1. [0.96666667 0.96666667 1.         0.93333333 0.96666667]
# 2. acc:  [0.95833333 1.         0.95833333 1.         0.875     ] 평균 :  0.9583
# model = KNeighborsClassifier()
# 1. acc:  [0.96666667 0.96666667 1.         0.9        0.96666667] 평균 :  0.96
# 2. acc:  [0.91666667 1.         0.95833333 1.         0.95833333] 평균 :  0.9667
# model = LogisticRegression()
# 1. acc:  [1.         0.96666667 1.         0.9        0.96666667] 평균 :  0.9667
# 2. acc:  [0.95833333 1.         0.95833333 1.         0.91666667] 평균 :  0.9667
# model = DecisionTreeClassifier()
# 1. acc:  [0.96666667 0.96666667 1.         0.9        0.93333333] 평균 :  0.9533
# 2. acc:  [0.95833333 0.95833333 0.95833333 1.         0.875     ] 평균 :  0.95
# model = DecisionTreeRegressor()
# 1. acc:  [0.94966443 0.95       1.         0.82824427 0.90338164] 평균 :  0.9263
# 2. acc:  [0.94392523 0.86956522 0.85       1.         0.82309582] 평균 :  0.8973
# model = RandomForestClassifier()
# 1. acc:  [0.9        0.96666667 1.         0.9        0.96666667] 평균 :  0.9467
# 2. acc:  [0.95833333 0.95833333 0.95833333 1.         0.875     ] 평균 :  0.95
model = RandomForestRegressor()
# 1. acc:  [0.95149664 0.94224    0.99950242 0.88052672 0.95813043] 평균 :  0.9464
# 2. acc:  [0.94377944 0.95481739 0.92518    0.99833162 0.8685543 ] 평균 :  0.9381

scores = cross_val_score(model, x_train, y_train, cv=kfold)
print("2. acc: " ,scores, "평균 : ",round(np.mean(scores), 4))
