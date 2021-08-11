from sklearn.svm import LinearSVC, SVC    # 애네가 먹히는지 확인
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression #  LogisticRegression 면접질문에서 많이나옴 (분류모델이라고 외우면됨)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

#실습 diabets
# 1. loss와 r2로 평가를 함
# 2. MinMax와 Standard 결과를 명시
from sklearn.metrics import accuracy_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.datasets import load_diabetes, load_iris,load_wine, load_boston
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=66)

n_splits=5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

parameters = [
    # {"n_estimators" : [100,200, 300]},  # 에포
    # {'max_depth' : [6, 8, 10, 12]},    # 깊이
    # {'min_samples_leaf' : [3,5,7,10]},
    # {'min_samples_split' : [2,3,5,10]},
    # {'n_jobs' : [-1,2,4,6]}   # cpu를 몇 개 쓸것인지  -1은 모든코어를 다쓰겠다.

    # {'randomforestregressor__min_samples_leaf' : [3, 5, 7], 'randomforestregressor__max_depth' : [2, 3, 5, 10]},
    # {'randomforestregressor__min_samples_split' : [6, 8, 10]},

    {'rg__min_samples_leaf' : [3, 5, 7], 'rg__max_depth' : [2, 3, 5, 10]},
    {'rg__min_samples_split' : [6, 8, 10]}, 

]
# pipe = make_pipeline(MinMaxScaler(), RandomForestRegressor())
pipe = Pipeline([("scaler", MinMaxScaler()), ("rg", RandomForestRegressor())])

model = RandomizedSearchCV(pipe, parameters, cv=kfold)
import time
start_time = time.time()
model.fit(x_train, y_train)

print("최적의 매개변수 : ", model.best_estimator_)
print("best_score : ", model.best_score_)
print("best_params : ", model.best_params_)


print("model.score : ", model.score(x_test, y_test))

y_predict = model.predict(x_test)
print("정답률 : ", r2_score(y_test, y_predict))
print("걸린시간 : ", time.time()- start_time)

# model.score :  0.8883155506822259
# 정답률 :  0.8883155506822259
# 걸린시간 :  5.304967880249023

# 최적의 매개변수 :  Pipeline(steps=[('scaler', MinMaxScaler()),
#                 ('rg', RandomForestRegressor(min_samples_split=6))])
# best_score :  0.8455600663297727
# best_params :  {'rg__min_samples_split': 6}
# model.score :  0.8830693491028088
# 정답률 :  0.8830693491028088
# 걸린시간 :  5.394445180892944