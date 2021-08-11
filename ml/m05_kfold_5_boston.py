from sklearn.svm import LinearSVC, SVC    # 애네가 먹히는지 확인
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression #  LogisticRegression 면접질문에서 많이나옴 (분류모델이라고 외우면됨)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import r2_score
import numpy as np

datasets = load_boston()
x = datasets.data
y = datasets.target


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=66)

n_splits=5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)


model = KNeighborsRegressor()
# acc:  [0.48564217 0.44681511 0.58468507 0.2899459  0.49101289] 평균 :  0.4596
# model = LinearRegression()
# acc:  [0.64520963 0.75944241 0.66980083 0.63214666 0.68836076] 평균 :  0.679
# model = DecisionTreeRegressor()
# acc:  [0.78441408 0.85199167 0.56955962 0.72819075 0.76352869] 평균 :  0.7395
# model = RandomForestRegressor()
# acc:  [0.83175546 0.88366608 0.77176307 0.85353444 0.86267072] 평균 :  0.8407

scores = cross_val_score(model, x_train, y_train, cv=kfold)
print("acc: " ,scores, "평균 : ",round(np.mean(scores), 4))