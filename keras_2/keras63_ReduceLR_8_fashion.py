from tensorflow.keras.datasets import fashion_mnist
import numpy as np
import matplotlib.pyplot as plt
from icecream import ic

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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, LSTM, Conv1D
model = Sequential()
# dnn
model.add(LSTM(128, activation='relu', input_shape=(28, 28), return_sequences=True))
model.add(Conv1D(64, 2, activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

# model.summary()


#3. 컴파일(ES), 훈련
from tensorflow.keras.optimizers import Adam, Nadam
optimizers = Adam(lr = 0.001)

model.compile(loss='categorical_crossentropy', optimizer=optimizers)

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
es = EarlyStopping(monitor='val_loss', patience=8, mode='min', verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, mode='auto', factor=0.5)

import time
start = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=300, validation_split=0.001, callbacks=[es, reduce_lr])
end = time.time() - start


#4. 평가, 예측
from sklearn.metrics import r2_score

y_predict = model.predict(x_test)
results = model.evaluate(x_test, y_test)
print('걸린시간 :', end)
r2 = r2_score(y_test, y_predict)
ic(r2)
# print('category :', results[0])
# print('accuracy :', results[1])

# 걸린시간 : 111.95567202568054
# category : 2.302614212036133
# accuracy : 0.10000000149011612

# Adam = 0.001
# 걸린시간 : 118.05785274505615
# ic| r2: 0.6862886716519957