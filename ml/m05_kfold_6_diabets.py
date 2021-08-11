from sklearn.svm import LinearSVC, SVC    # 애네가 먹히는지 확인
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression #  LogisticRegression 면접질문에서 많이나옴 (분류모델이라고 외우면됨)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

#실습 diabets
# 1. loss와 r2로 평가를 함
# 2. MinMax와 Standard 결과를 명시

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import r2_score

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=66)

n_splits=5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

model = KNeighborsRegressor()
# acc:  [0.46856904 0.2229516  0.46472072 0.50694267 0.27327008] 평균 :  0.3873
# model = LogisticRegression()
# acc:  [0.         0.01612903 0.         0.01612903 0.        ] 평균 :  0.0065
# model = LinearRegression()
# acc:  [0.44291401 0.32335125 0.54104637 0.56657751 0.39383584] 평균 :  0.4535
# model = DecisionTreeRegressor()
# acc:  [ 0.23269303 -0.20800609  0.28735727  0.25995761 -0.40065033] 평균 :  0.0343
# model = RandomForestRegressor()
# acc:  [0.42822286 0.27815729 0.53104358 0.48116099 0.37833755] 평균 :  0.4194
scores = cross_val_score(model, x_train, y_train, cv=kfold)
print("acc: " ,scores, "평균 : ",round(np.mean(scores), 4))