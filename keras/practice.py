from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np

datasets = load_boston()
x = datasets.data
y = datasets.target
# print(x.shape)
# print(y.shape)
# print(datasets.feature_names)
# print(datasets.DESCR)
print(np.min(x), np.max(x))  # 0.0 711.0

#데이터 전처리
# x = x/711.
# x = (x - np.min(x)) / (np.max(x) - np.min(x))
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=66)
from sklearn.preprocessing import PowerTransformer, StandardScaler
# scaler = PowerTransformer()
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test) 
# print(x_scale[:])
# print(np.min(x_scale), np.max(x_scale))



model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(13,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=8)

loss = model.evaluate(x_test, y_test)
print('lose : ', loss)
y_predict = model.predict(x_test)
print('예측: ', y_predict)
r2 = r2_score(y_test, y_predict)
print('r2socre: ', r2)

# r2socre:  0.9119163769748695

# StandardScaler
# r2socre:  0.9017531437212003