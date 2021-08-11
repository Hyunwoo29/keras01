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

x = datasets.data
y = datasets.target

from sklearn.preprocessing import MinMaxScaler, StandardScaler


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)
from sklearn.pipeline import make_pipeline, Pipeline
model = make_pipeline(MinMaxScaler(), RandomForestClassifier())


from sklearn.metrics import r2_score

model.fit(x_train,y_train)

import time
start_time = time.time()
model.fit(x_train, y_train)

# print("최적의 매개변수 : ", model.best_estimator_)
# print("best_score : ", model.best_score_)
# print("best_params : ", model.best_params_)

print("model.score : ", model.score(x_test, y_test))

y_predict = model.predict(x_test)
print("정답률 : ", r2_score(y_test, y_predict))
print("걸린시간 : ", time.time()- start_time)

# model.score :  1.0
# 정답률 :  1.0
# 걸린시간 :  0.09474706649780273