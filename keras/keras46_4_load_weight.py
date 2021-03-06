from enum import auto
from os import scandir
from sklearn.preprocessing import MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer, StandardScaler, MinMaxScaler, OneHotEncoder
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
# import pandas as pd
from sklearn.datasets import load_diabetes
from icecream import ic
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPool2D
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time

# save파일명: keras46_1_save_model.h5


#1. 데이터
datasets = load_diabetes()

x = datasets.data
y = datasets.target

ic(x.shape, y.shape) # x.shape: (442, 10), y.shape: (442,)

# ic(datasets.feature_names)
# # ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
# ic(datasets.DESCR)

# ic(y[:30]) # 30개 데이터 출력

# ic(np.min(y), np.max(y)) # 최소, 최대값 

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.7, random_state=60)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델 구성

model = Sequential()
model.add(Dense(256, input_shape=(10,)))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))

# model.save('./_save/keras46_1_save_model_1.h5')
# model.save_weights('./_save/keras46_1_save_weights_1.h5')

# model = load_model('./_save/keras46_1_save_weights_1.h5') # ValueError: No model found in config file.
# model = load_model('./_save/keras46_1_save_weights_2.h5') # No model found in config file.

# 결론: save_weights에는 모델 저장 X, load_weights는 모델이 구성되어있어야함 


#model = load_model('./_save/keras46_1_save_model_1.h5') # Total params: 48,193 모델만 저장 됐기때문에 컴파일과 핏을 해야함(모델만 저장하고싶은 경우 모델 밑에다 세이브 모델)
#model = load_model('./_save/keras46_1_save_model_2.h5') #모델, 핏, 컴파일 저장했기 때문에 결과가 바로 나옴(가중치까지 저장하고싶은경우 컴파일과 핏 다음) Total params: 48,193

model.summary()



#3. 컴파일, 훈련
es = EarlyStopping(monitor='val_loss', patience=20, mode='auto')
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
#model.load_weights('./_save/keras46_1_save_weights_1.h5') # load_weights할땐 순수하게 w만 들어감
# ic| loss: [28505.87109375, 149.60935974121094]
# ic| r2: -3.655426768090252
model.load_weights('./_save/keras46_1_save_weights_2.h5') # 제대로 된 W불러옴 (컴파일, 훈련 다음에 save_weights한것)
# ic| loss: [2920.5185546875, 43.422550201416016]
# ic| r2: 0.5230365098783467
start = time.time()
# model.fit(x_train, y_train, epochs=500, verbose=2, batch_size=8, validation_split=0.2, shuffle=True, callbacks=[es])
걸린시간 = round((time.time() - start) /60,1)

# model.save('./_save/keras46_1_save_model_2.h5')
# model.save_weights('./_save/keras46_1_save_weights_2.h5')


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
ic(loss)

y_predict = model.predict(x_test)
# ic(y_predict)

r2 = r2_score(y_test, y_predict)
ic(r2)

ic(걸린시간)