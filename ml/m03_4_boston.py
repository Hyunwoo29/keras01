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
# print(x.shape)
# print(y.shape)
# print(datasets.feature_names)
# print(datasets.DESCR)
# print(np.min(x), np.max(x))  # 0.0 711.0

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=66)

#데이터 전처리
# x = x/711.
# x = (x - np.min(x)) / (np.max(x) - np.min(x))

from sklearn.preprocessing import MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer

scaler = MaxAbsScaler()
# scaler = RobustScaler()
# scaler = QuantileTransformer()
# scaler = PowerTransformer()
scaler.fit(x_train)
x_train = scaler.transform(x_train) # train은 훈련을 시키고, test는 훈련에 포함되면 안된다.
x_test = scaler.transform(x_test)  # 왜냐하면 train은 minmaxcaler가 0~1 이고 test는 0~ 1.2 범위를 넘을 수 있기때문에


# model = LinearSVC()
# model = SVC()
# model = KNeighborsClassifier()
# model = LogisticRegression()
# model = DecisionTreeClassifier()
# model = RandomForestClassifier()

# model = LinearRegression()
# modle.score :  0.8044888426543627
# model = DecisionTreeRegressor()
# modle.score :  0.6316560051841551
# model = KNeighborsRegressor()
# modle.score :  0.7843808845761827
# model = RandomForestRegressor()
# modle.score :  0.8892545396030381



model.fit(x_train, y_train)

results = model.score(x_test, y_test) # 머신러닝에서는 evaluate 개념이 score이다.
print("modle.score : ", results)

