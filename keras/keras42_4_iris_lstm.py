import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# 1. data
datasets = load_iris()

# print(datasets.DESCR)
# print(datasets.feature_names) # ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']     

x = datasets.data # (150, 4)
y = datasets.target # (150, ) - incoding-> (150,3)

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)

x_train, x_test, y_train, y_test = train_test_split(x, y,
      test_size=0.25, shuffle=True, random_state=66)

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

input1 = Input(shape=(4, 1))
xxx = LSTM(units=20, activation='relu')(input1)
xxx = Dense(128, activation='relu')(xxx)
xxx = Dense(64, activation='relu')(xxx)
xxx = Dense(32, activation='relu')(xxx)
xxx = Dense(16, activation='relu')(xxx)
xxx = Dense(8, activation='relu')(xxx)
output1 = Dense(3, activation='softmax')(xxx)

model = Model(inputs=input1, outputs=output1)

# 3. 컴파일 훈련
# data 형태가 다르므로 mse 대신 categorical_crossentropy 사용
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', patience=15, mode='min', verbose=1)

import time 
start_time = time.time()
model.fit(x_train, y_train, epochs=1000, batch_size=16, verbose=2,
    validation_split=0.2, callbacks=[es])
end_time = time.time() - start_time

# 4. 평가 예측

loss = model.evaluate(x_test, y_test)
print("time : ", end_time)
print('loss : ', loss[0])
print('acc : ', loss[1])


# time :  6.312670469284058
# loss :  0.15171565115451813
# acc :  0.9473684430122375
