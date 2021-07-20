from tensorflow.keras.datasets import cifar10

import numpy as np

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.reshape(50000, 32 * 32 * 3)
x_test = x_test.reshape(10000, 32 * 32 * 3)


# print(np.unique(y_train)) # [0 1 2 3 4 5 6 7 8 9]
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer
scaler = StandardScaler()
# scaler.fit(x_train) 
# x_train = scaler.transform(x_train) 
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test) 

from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder()
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
one.fit(y_train)
y_train = one.transform(y_train).toarray() # (50000, 10)

y_test = one.transform(y_test).toarray() # (10000, 10)
#2. 모델

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout

model = Sequential()
model.add(Dense(120, activation='relu', input_shape=(32*32*3,)))
model.add(Dropout(0.1))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

# 컴파일, 훈련 matrics=['acc']
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='accuracy') # 2진 분류에서는 binary_crossentropy 를 쓰면된다.

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=8, mode='min', verbose=1)

model.fit(x_train, y_train, epochs=100, batch_size=300, validation_split=0.02, callbacks=[es])

loss = model.evaluate(x_test, y_test)     
print('lose : ', loss[0])
print('accuracy : ', loss[1])

# lose :  1.3691636323928833
# accuracy :  0.5217999815940857