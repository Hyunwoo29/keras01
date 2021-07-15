from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
import matplotlib as mpl
from matplotlib import rc

datasets = load_boston()
x = datasets.data
y = datasets.target
# print(x.shape)
# print(y.shape)
# print(datasets.feature_names)
# print(datasets.DESCR)
print(np.min(x), np.max(x))  # 0.0 711.0

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=66)

#데이터 전처리
# x = x/711.
# x = (x - np.min(x)) / (np.max(x) - np.min(x))

from sklearn.preprocessing import MinMaxScaler, StandardScaler
# scaler = MinMaxScaler()
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) # train은 훈련을 시키고, test는 훈련에 포함되면 안된다.
x_test = scaler.transform(x_test)  # 왜냐하면 train은 minmaxcaler가 0~1 이고 test는 0~ 1.2 범위를 넘을 수 있기때문에






#모델구성
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(13,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=5, mode='min', verbose=1) # patience는 몇번까지 참을거냐 이뜻. # mode min은 patience가 5라면 거기서 부터 멈춰서 다시


hist = model.fit(x_train, y_train, epochs=100, batch_size=8, validation_split=0.2, callbacks=[es])

print(hist) # <tensorflow.python.keras.callbacks.History object at 0x00000172242EF130>

print(hist.history.keys()) # dict_keys(['loss', 'val_loss'])
print(hist.history['loss'])
print(hist.history['val_loss'])
# 발로스를 더 많이쓸거다. 이유는 로스가 값이 더 좋게나와서 좋은걸로 지표를 잡으면
# 안되구 발로스가 값이 더 안좋아서 안좋은걸로 지표를 잡아 방향성을 잡아야한다.



loss = model.evaluate(x_test, y_test)
print('lose : ', loss)
y_predict = model.predict(x_test)
print('예측: ', y_predict)
r2 = r2_score(y_test, y_predict)
print('r2socre: ', r2)

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'gulim'  # 한글깨짐 폰트설정
plt.plot(hist.history['loss']) # x: epoch / y : hist.history['loss]
plt.plot(hist.history['val_loss'])

# plt.title("loss, val_loss")
plt.title('로스, 발로스')
plt.xlabel('epochs')
plt.ylabel('loss, val_loss')
plt.legend(['train_loss', 'val loss'])
plt.show()