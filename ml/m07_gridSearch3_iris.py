# 실습
# 모델 : RandomForestClassifier

import numpy as np
from sklearn import datasets
from sklearn.datasets import load_iris
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression #  LogisticRegression 면접질문에서 많이나옴 (분류모델이라고 외우면됨)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score

datasets = load_iris()

x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)
n_splits=5
kfold = KFold(n_splits=n_splits, shuffle=True)

parameters = [
    {"n_estimators" : [100,200]},  
    {'max_depth' : [6, 8, 10, 12]},    # 깊이
    {'min_samples_leaf' : [3,5,7,10]},
    {'min_samples_split' : [2,3,5,10]},
    {'n_jobs' : [-1,2,4]}   # cpu를 몇 개 쓸것인지
]

model = GridSearchCV(RandomForestClassifier(), parameters, cv=kfold)

model.fit(x_train, y_train)

# print("최적의 매개변수 : ", model.best_estimator_)
# print("best_score : ", model.best_score_)
print("model.score : ", model.score(x_test, y_test))

y_predict = model.predict(x_test)
print("정답률 : ", accuracy_score(y_test, y_predict))


# model.score :  0.9666666666666667
# 정답률 :  0.9666666666666667