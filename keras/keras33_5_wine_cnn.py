import numpy as np
from sklearn import datasets
from sklearn.datasets import load_wine
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
datasets = load_wine()
# print(datasets.DESCR)
# print(datasets.feature_names)

x = datasets.data
y = datasets.target

print(x.shape, y.shape)
print(y)

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=66)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
# scaler = MinMaxScaler()
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test) 

model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(13,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(3, activation='softmax')) # 다중분류


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='accuracy') # 2진 분류에서는 binary_crossentropy 를 쓰면된다.

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', patience=5, mode='min', verbose=1)

model.fit(x_train, y_train, epochs=100, batch_size=8, validation_split=0.2, callbacks=[es])

loss = model.evaluate(x_test, y_test)     
print('lose : ', loss[0])
print('accuracy : ', loss[1])

print("=========== 예측 =============")
print(y_test)
y_predict = model.predict(x_test)
print(y_predict)

#완성하시오!!
# acc 0.8 이상 만들것!

# lose :  0.03139385208487511
# accuracy :  0.9814814925193787