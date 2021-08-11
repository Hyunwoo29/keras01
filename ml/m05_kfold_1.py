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

n_splits=5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)


# model = LinearSVC()
# [0.96666667 0.96666667 1.         0.9        1.        ]
# model = SVC()
# [0.96666667 0.96666667 1.         0.93333333 0.96666667]
# model = KNeighborsClassifier()
# acc:  [0.96666667 0.96666667 1.         0.9        0.96666667] 평균 :  0.96
# model = LogisticRegression()
# acc:  [1.         0.96666667 1.         0.9        0.96666667] 평균 :  0.9667
# model = DecisionTreeClassifier()
# acc:  [0.96666667 0.96666667 1.         0.9        0.93333333] 평균 :  0.9533
# model = DecisionTreeRegressor()
# acc:  [0.94966443 0.95       1.         0.82824427 0.90338164] 평균 :  0.9263
# model = RandomForestClassifier()
# acc:  [0.9        0.96666667 1.         0.9        0.96666667] 평균 :  0.9467
model = RandomForestRegressor()
# acc:  [0.95149664 0.94224    0.99950242 0.88052672 0.95813043] 평균 :  0.9464

scores = cross_val_score(model, x, y, cv=kfold)
print("acc: " ,scores, "평균 : ",round(np.mean(scores), 4))
