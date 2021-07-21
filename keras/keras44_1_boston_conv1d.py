from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, LSTM, Conv1D
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
from icecream import ic
import time

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

from sklearn.preprocessing import MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer, StandardScaler

scaler = StandardScaler()
# scaler = PowerTransformer()
# scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1) # x_train.shape: (354, 13, 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1) # x_test.shape: (152, 13, 1)

# ic(x_train.shape, x_test.shape)


# ic(np.unique(y))






#모델구성
model = Sequential()
model.add(LSTM(20, activation='relu', input_shape=(13,1), return_sequences=True))
model.add(Conv1D(128, 2))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Flatten())
model.add(Dense(1))

#컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping
model.compile(loss='mse', optimizer='adam')

es =EarlyStopping(monitor='val_loss', patience=20, mode='auto', verbose=1) 
start = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=8, verbose=1, validation_split=0.2, callbacks=[es])
시간 = round((time.time() - start) /60,1)

loss = model.evaluate(x_test, y_test)
print('lose : ', loss)
y_predict = model.predict(x_test)
# print('예측: ', y_predict)
r2 = r2_score(y_test, y_predict)
print('r2socre: ', r2)
ic(f'{시간}')

# lose :  12.406935691833496
# r2socre:  0.8498260563754094
# ic| f'{시간}': '1.7'