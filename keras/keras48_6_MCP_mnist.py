from sklearn.preprocessing import MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer, StandardScaler, MinMaxScaler, OneHotEncoder
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from tensorflow.keras.datasets import mnist
from icecream import ic
import time
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout, MaxPool2D, LSTM 
from tensorflow.keras.models import Sequential, load_model



#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data() # x_train.shape: (60000, 28, 28), y_train.shape: (10000, 28, 28)

x_train = x_train.reshape(60000, 28 * 28)
x_test = x_test.reshape(10000, 28 * 28)


# 전처리 

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train) 
x_test = scaler.transform(x_test)

x_train = x_train.reshape(60000, 28 * 28 , 1)
x_test = x_test.reshape(10000, 28 * 28 , 1)


# ic(np.unique(y_train))

one = OneHotEncoder()
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

one.fit(y_train)
y_train = one.transform(y_train).toarray()
y_test = one.transform(y_test).toarray()


# model = Sequential()
# model.add(LSTM(40, activation='relu', input_shape=(28*28,1), return_sequences=True))
# model.add(Conv1D(128, 2))
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(16, activation='relu'))
# model.add(Flatten())
# model.add(Dense(10, activation='softmax'))

# model.summary()

# #3. 컴파일, 훈련
# es = EarlyStopping(monitor='acc', patience=3, mode='auto', verbose=1)
# cp = ModelCheckpoint(monitor='val_loss', patience=3, mode='auto', save_best_only=True, filepath='./_save/ModelCheckPoint/keras48_6_MCP_mnist.hdf5')
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
# start = time.time()
# model.fit(x_train, y_train, epochs=10, verbose=1, validation_split=0.2, batch_size=1024, shuffle=True, callbacks=[es, cp])
# 걸린시간 = round((time.time() - start) /60,1)

# model.save('./_save/ModelCheckPoint/keras48_6_model_save_mnist.h5')

# model = load_model('./_save/ModelCheckPoint/keras48_6_model_save_mnist.h5')
model = load_model('./_save/ModelCheckPoint/keras48_6_MCP_mnist.hdf5')

#4. 평가, 예측

y_predict = model.predict(x_test)
loss = model.evaluate(x_test, y_test)
ic(loss[0])
ic(loss[1])
# ic(f'{걸린시간}분')


#모델 체크포인트 세이브
# ic| loss[0]: nan
# ic| loss[1]: 0.09799999743700027

#모델 로드
# ic| loss[0]: nan
# ic| loss[1]: 0.09799999743700027

#체크포인트 로드
# ic| loss[0]: 7.42052393739645e+33
# ic| loss[1]: 0.9549999833106995