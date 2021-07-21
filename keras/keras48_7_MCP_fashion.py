from tensorflow.keras.datasets import fashion_mnist
import numpy as np
import matplotlib.pyplot as plt
from icecream import ic
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout, MaxPool2D, LSTM 
from tensorflow.keras.models import Sequential, load_model
### cnn -> dnn

#1. 데이터
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
# print(x_train.shape, y_train.shape)   # (60000, 28, 28) (60000,)
# print(x_test.shape, y_test.shape)     # (10000, 28, 28) (10000,)

x_train = x_train.reshape(60000, 28, 28)
x_test = x_test.reshape(10000, 28, 28)

ic(np.unique(y_train))   # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

#1-2. 데이터 전처리
from sklearn.preprocessing import OneHotEncoder
onehot = OneHotEncoder()
onehot.fit(y_train)
y_train = onehot.transform(y_train).toarray()  # (60000, 10
y_test = onehot.transform(y_test).toarray()   # (10000, 10)
ic(y_train.shape, y_test.shape)


#2. 모델 구성
# model = Sequential()
# # dnn
# model.add(LSTM(128, activation='relu', input_shape=(28, 28), return_sequences=True))
# model.add(Conv1D(64, 2, activation='relu'))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(10, activation='softmax'))

# model.summary()

import time
#3. 컴파일(ES), 훈련
# es = EarlyStopping(monitor='acc', patience=3, mode='auto', verbose=1)
# cp = ModelCheckpoint(monitor='val_loss', patience=3, mode='auto', save_best_only=True, filepath='./_save/ModelCheckPoint/keras48_6_MCP_mnist.hdf5')
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
# start = time.time()
# model.fit(x_train, y_train, epochs=10, verbose=1, validation_split=0.2, batch_size=1024, shuffle=True, callbacks=[es, cp])
# end = round((time.time() - start) /60,1)
# from tensorflow.keras.callbacks import EarlyStopping
# es = EarlyStopping(monitor='val_loss', patience=8, mode='min', verbose=1)


# model.save('./_save/ModelCheckPoint/keras48_7_model_save_fashion_mnist.h5')

# model = load_model('./_save/ModelCheckPoint/keras48_7_model_save_fashion_mnist.h5')
model = load_model('./_save/ModelCheckPoint/keras48_7_MCP_fashion_mnist.hdf5')

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
# print('걸린시간 :', end)
print('category :', results[0])
print('accuracy :', results[1])


#모델 체크포인트 세이브
# 걸린시간 : 0.4
# category : 1.095536470413208
# accuracy : 0.6035000085830688

#모델 로드
# category : 1.095536470413208
# accuracy : 0.6035000085830688

#체크포인트 로드
