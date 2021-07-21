from sklearn.preprocessing import MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer, StandardScaler, MinMaxScaler, OneHotEncoder
import numpy as np
from tensorflow.keras.datasets import cifar10
from icecream import ic
import time
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout, MaxPool2D, LSTM 
from tensorflow.keras.models import Sequential, load_model

# 1. 데이터
(x_train,y_train), (x_test, y_test) = cifar10.load_data()
# ic(x_train.shape, y_train.shape)   # (50000, 32, 32, 3), (50000, 1)
# ic(x_test.shape, y_test.shape)     # (10000, 32, 32, 3), (10000, 1)

x_train = x_train.reshape(50000, 32, 96)
x_test = x_test.reshape(10000, 32, 96)

ic(np.unique(y_train))   # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] - 10개
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

# 1-2. 데이터전처리
from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder()
one.fit(y_train)
y_train = one.transform(y_train).toarray()
y_test = one.transform(y_test).toarray()
# ic(y_train.shape, y_test.shape)   # (50000, 10), (10000, 10)

# 2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, LSTM, Conv1D

# model = Sequential()
# dnn
# model.add(LSTM(128, activation='relu', input_shape=(32, 96), return_sequences=True))
# model.add(Conv1D(64, 2, activation='relu'))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(10, activation='softmax'))

# model.summary()


# 3. 컴파일(ES), 훈련
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='accuracy')

# from tensorflow.keras.callbacks import EarlyStopping
# es = EarlyStopping(monitor='val_loss', mode='auto', patience=10, verbose=1)
# cp = ModelCheckpoint(monitor='val_loss', patience=3, mode='auto', save_best_only=True, filepath='./_save/ModelCheckPoint/keras48_8_MCP_cifar10.hdf5')
# import time
# start = time.time()
# model.fit(x_train, y_train, epochs=5, verbose=1, validation_split=0.2, batch_size=1024, shuffle=True, callbacks=[es, cp])
# end = time.time() - start

# model.save('./_save/ModelCheckPoint/keras48_8_model_save_cifar10.h5')

# model = load_model('./_save/ModelCheckPoint/keras48_8_model_save_cifar10.h5')
model = load_model('./_save/ModelCheckPoint/keras48_8_MCP_cifar10.hdf5')
# 4. 평가, 예측
results = model.evaluate(x_test, y_test)
# print('걸린시간 :', end)
print('category :', results[0])
print('accuracy :', results[1])

# 걸린시간 : 229.66736102104187
# category : 2.3025877475738525
# accuracy : 0.10000000149011612

#모델 체크포인트 세이브
# 걸린시간 : 15.439188003540039
# category : 2.30260968208313
# accuracy : 0.10000000149011612

#모델 로드
# category : 2.30260968208313
# accuracy : 0.10000000149011612

#체크포인트 로드
# category : 2.30260968208313
# accuracy : 0.10000000149011612