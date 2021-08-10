import numpy as np
from sklearn import datasets
from sklearn.datasets import load_iris
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split


datasets = load_iris()
print(datasets.DESCR)
print(datasets.feature_names)

x = datasets.data
y = datasets.target


# from tensorflow.keras.utils import to_categorical
# y = to_categorical(y)
# print(y[:5])

# print(y.shape) # (150, 3)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=66)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
# scaler = MinMaxScaler()
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test) 

from sklearn.svm import LinearSVC
model = LinearSVC()

# model = Sequential()
# model.add(Dense(128, activation='relu', input_shape=(4,)))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(3, activation='softmax')) # 다중분류

# 훈련
model.fit(x_train, y_train)


# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='accuracy') # 2진 분류에서는 binary_crossentropy 를 쓰면된다.

# from tensorflow.keras.callbacks import EarlyStopping
# es = EarlyStopping(monitor='loss', patience=5, mode='min', verbose=1)

# model.fit(x_train, y_train, epochs=100, batch_size=8, validation_split=0.2, callbacks=[es])

# 평가, 예측
results = model.score(x_test, y_test) # 머신러닝에서는 evaluate 개념이 score이다.
print("modle.score : ", results)


# loss = model.evaluate(x_test, y_test)     
# print('lose : ', loss[0])
# print('accuracy : ', loss[1])
from sklearn.metrics import r2_score, accuracy_score
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print("accuracy_score : ", acc)

print("=========== 예측 =============")
print(y_test[:5])
y_predict2 = model.predict(x_test[:5])
print(y_predict2)

# r2 = r2_score(y_test, y_predict)
# print('r2score: ', r2)