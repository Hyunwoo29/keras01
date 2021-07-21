#overfit을 극복하자
from sklearn.preprocessing import MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer, StandardScaler, MinMaxScaler, OneHotEncoder
import numpy as np
from tensorflow.keras.datasets import cifar100
from icecream import ic
import time
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout, MaxPool2D, LSTM 
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.utils import to_categorical

### GlobalAveragePooling2D

# 1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

x_train = x_train.reshape(50000, 32 * 32 * 3)
x_test = x_test.reshape(10000, 32 * 32 * 3)
print(np.unique(y_train)) 
# 전처리 하기 -> scailing
# 단, 2차원 데이터만 가능하므로 4차원 -> 2차원
# x_train = x_train/255.
# x_test = x_test/255.
print(x_train.shape, x_test.shape) # (50000, 3072) (10000, 3072)

# 1-2. x 데이터 전처리
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
x_train = x_train.reshape(50000, 32 , 96)
x_test = x_test.reshape(10000, 32, 96)

# 1-3. y 데이터 전처리 -> one-hot-encoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape, y_test.shape)


# 2. 모델 구성(GlobalAveragePooling2D 사용)
# model = Sequential()
# model.add(LSTM(128, input_shape=(32, 96), activation='relu', return_sequences=True))
# model.add(Conv1D(64, 2, activation='relu'))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))    
# model.add(Dense(64, activation='relu')) 
# model.add(Dense(32, activation='relu'))
# model.add(Dense(100, activation='softmax'))


# # 3. 컴파일(ES), 훈련
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
# es = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')
# cp = ModelCheckpoint(monitor='val_loss', patience=3, mode='auto', save_best_only=True, filepath='./_save/ModelCheckPoint/keras48_9_MCP_cifar100.hdf5')
# start_time = time.time()
# hist = model.fit(x_train, y_train, epochs=100, verbose=1, callbacks=[es,cp], validation_split=0.2, shuffle=True, batch_size=512)
# end_time = time.time() - start_time

# model.save('./_save/ModelCheckPoint/keras48_9_model_save_cifar100.h5')

# model = load_model('./_save/ModelCheckPoint/keras48_9_model_save_cifar100.h5')
model = load_model('./_save/ModelCheckPoint/keras48_9_MCP_cifar100.hdf5')

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
# print("걸린시간 :", end_time)
print('category :', loss[0])
print('accuracy :', loss[1])


#모델 체크포인트 세이브
# 걸린시간 : 104.19983339309692
# category : 3.515530824661255
# accuracy : 0.26249998807907104

#모델 로드
# category : 3.515530824661255
# accuracy : 0.26249998807907104

#체크포인트 로드
# category : 3.09340763092041
# accuracy : 0.2574999928474426