from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from tensorflow.python.keras.layers.core import Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

# print(x.shape, y.shape) # (442, 10) (442,)

# print(datasets.feature_names) # ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
# print(datasets.DESCR)

# print(y[:30])
# print(np.min(y), np.max(y))
#2. 모델구성
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True, random_state=9)

model = Sequential()
model.add(Dense(480, input_shape=(10,), activation='relu'))
model.add(Dense(140, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=33, validation_split=0.1, shuffle=True)

loss = model.evaluate(x_test, y_test)
print('lose : ', loss)
y_predict = model.predict(x_test)
print('예측: ', y_predict)
r2 = r2_score(y_test, y_predict)
print('r2socre: ', r2)

#3. 컴파일, 훈련

#4. 평가, 예측