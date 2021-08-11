# 실습, 모델구성하고 완료
# 회귀 데이터를  Classifier로 만들었을 경우에 에러 확인!

from sklearn.svm import LinearSVC, SVC    # 애네가 먹히는지 확인
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression #  LogisticRegression 면접질문에서 많이나옴 (분류모델이라고 외우면됨)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np

datasets = load_boston()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=66)



from sklearn.preprocessing import MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer, MinMaxScaler

from sklearn.pipeline import make_pipeline, Pipeline
model = make_pipeline(MinMaxScaler(), RandomForestRegressor())


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

# model.score :  0.8950021365868397
# 정답률 :  0.8950021365868397
# 걸린시간 :  0.16556215286254883