import numpy as np
from sklearn import datasets
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# 1. data 구성
datasets = load_diabetes()

x = datasets.data # (442, 10)
y = datasets.target # (442,)

x_train, x_test, y_train, y_test = train_test_split(x, y,
      test_size=0.25, shuffle=True, random_state=9)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer
scaler = MinMaxScaler()
scaler.fit(x_train) 
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test) 

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

# 2. model 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input

input1 = Input(shape=(10, 1))
xxx = LSTM(units=20, activation='relu')(input1)
xxx = Dense(128, activation='relu')(xxx)
xxx = Dense(64, activation='relu')(xxx)
xxx = Dense(32, activation='relu')(xxx)
xxx = Dense(16, activation='relu')(xxx)
xxx = Dense(8, activation='relu')(xxx)
output1 = Dense(1)(xxx)

model = Model(inputs=input1, outputs=output1)

# 3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=30, mode='min', verbose=1)

import time 
start_time = time.time()
model.fit(x_train, y_train, epochs=500, batch_size=16, verbose=1,
    validation_split=0.2, callbacks=[es])
end_time = time.time() - start_time

# 4. 평가 예측
y_predict = model.predict([x_test])
# print('x의 예측값 : ', y_predict)

loss = model.evaluate(x_test, y_test)
print("time : ", end_time)
print('loss : ', loss)
r2 = r2_score(y_test, y_predict)
print('R^2 score : ', r2)

# time :  111.28265428543091
# loss :  2830.69873046875
# R^2 score :  0.5093729878281273