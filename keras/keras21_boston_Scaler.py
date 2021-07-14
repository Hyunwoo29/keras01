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
print(np.min(x), np.max(x))  # 0.0 711.0

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






#모델구성
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(13,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=8)

loss = model.evaluate(x_test, y_test)
print('lose : ', loss)
y_predict = model.predict(x_test)
# print('예측: ', y_predict)
r2 = r2_score(y_test, y_predict)
print('r2socre: ', r2)


# MaxAbsScaler   --> 최대절대값과 0이 각각 1, 0이 되도록 스케일링, 큰 이상치에 민감할 수 있다.
# lose :  9.004426956176758
# r2socre:  0.8910101307874165

# lose :  8.008648872375488
# r2socre:  0.9030630733714031

# lose :  7.1432271003723145
# r2socre:  0.9135381576662176

# RobustScaler ---> 중앙값과 IQR사용. 아웃라이어의 영향을 최소화, IQR = Q3 - Q1 : 즉, 25퍼센타일과 75퍼센타일의 값들을 다룬다.
# lose :  8.188029289245605
# r2socre:  0.9008918395617806

# lose :  7.699021339416504
# r2socre:  0.9068107970226118

# QuantileTransformer  ---->기본적으로 1000개 분위를 사용하여 데이터를 '균등분포' 시킵니다. 
# Robust처럼 이상치에 민감X, 0~1사이로 압축합니다.
# lose :  13.593234062194824
# r2socre:  0.835467063802032

# lose :  12.051332473754883
# r2socre:  0.8541302928415968

# lose :  8.959607124328613
# r2socre:  0.8915526219263451

# PowerTransformer -----> 데이터의 특성별로 정규분포형태에 가깝도록 변환
# lose :  11.664231300354004
# r2socre:  0.8588157776453507

# lose :  10.108380317687988
# r2socre:  0.8776478536856895
