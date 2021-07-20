import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
#데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 전처리 해야함
# x_train = x_train.reshape(60000, 28*28)
# x_test = x_test.reshape(10000, 28*28)
x_train = x_train.reshape(60000, 28 * 28, 1)
x_test = x_test.reshape(10000, 28 * 28, 1)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, PowerTransformer
# scaler = MinMaxScaler()
# scaler = PowerTransformer()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# x 데이터 전처리
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

# y 데이터 전처리
# from sklearn.preprocessing import OneHotEncoder
# one = OneHotEncoder()
# y_train = y_train.reshape(-1,1)
# y_test = y_test.reshape(-1,1)
# one.fit(y_train)
# y_train = one.transform(y_train).toarray() # (60000, 10)
# y_test = one.transform(y_test).toarray() # (10000, 10)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
# #2. 모델

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, GlobalAveragePooling1D
#dnn
model = Sequential()
# model.add(Dense(100, input_shape=(28*28,)))
model.add(Dense(100, activation='relu', input_shape=(28 * 28,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))


# cnn
# model = Sequential()
# model.add(Conv2D(filters=100, kernel_size=(4,4), padding='same', input_shape=(28,28,1)))
# model.add(Conv2D(128, (2, 2), padding='valid', activation='relu'))
# model.add(Conv2D(64, (2, 2), padding='valid', activation='relu'))
# model.add(Conv2D(64, (2, 2), padding='valid', activation='relu'))
# model.add(Flatten())
# model.add(Conv2D(32, (2, 2), padding='valid', activation='relu'))
# model.add(GlobalAveragePooling1D())
# model.add(Dense(10, activation='softmax'))

# 컴파일, 훈련 matrics=['acc']
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='accuracy') # 2진 분류에서는 binary_crossentropy 를 쓰면된다.

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=8, mode='min', verbose=1)

import time
start_time = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=300, validation_split=0.02, callbacks=[es])
end_time = time.time() - start_time
loss = model.evaluate(x_test, y_test)  
print("time = ", end_time)   
print('lose : ', loss[0])
print('accuracy : ', loss[1])

# dnn
# time =  16.880542993545532
# lose :  0.1235533058643341
# accuracy :  0.9708999991416931

