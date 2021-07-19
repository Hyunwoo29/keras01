import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
#데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 전처리 해야함
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)


# print(np.unique(y_train)) # [0 1 2 3 4 5 6 7 8 9]

from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder()
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
one.fit(y_train)
y_train = one.transform(y_train).toarray() # (60000, 10)
y_test = one.transform(y_test).toarray() # (10000, 10)
#2. 모델

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D

model = Sequential()
model.add(Dense(100, input_shape=(28*28,)))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))

model.add(Dense(10, activation='softmax'))

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

# 평가 , 예측 predict 할 필요 없다

# acc로만 판단해보자

#만들어야함!
# 0.98 이상 선착순 3명 커피

# lose :  0.04199787974357605
# accuracy :  0.9900000095367432

# lose :  0.03559916466474533
# accuracy :  0.9912999868392944