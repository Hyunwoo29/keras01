from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
datasets = load_boston()
x = datasets.data
y = datasets.target



print(x.shape)
print(y.shape)

print(datasets.feature_names)

print(datasets.DESCR)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, shuffle=True, random_state=65)


model = Sequential()
model.add(Dense(15, inpurt_dim=13))
model.add(Dense(22))
model.add(Dense(22))
model.add(Dense(22))
model.add(Dense(22))
model.add(Dense(22))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1)

y_predict = model.predict([x_test])
print('x의 예측값: ', y_predict)

loss = model.evaluate(x_test, y_test)
print('loss: ', loss)

r2 = r2_score(y_test, y_predict)
print('r2score: ', r2)
