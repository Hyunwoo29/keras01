from tensorflow.keras.datasets import cifar100

import numpy as np
from tensorflow.python.keras.layers.core import Dropout

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

x_train = x_train.reshape(50000, 32, 32, 3)/255.0
x_test = x_test.reshape(10000, 32, 32, 3)/255.0




from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder()
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
one.fit(y_train)
y_train = one.transform(y_train).toarray() # (50000, 100)

y_test = one.transform(y_test).toarray() # (10000, 100)
#2. 모델

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D

model = Sequential()
model.add(Conv2D(filters=100, kernel_size=(4,4), padding='same', input_shape=(32, 32, 3)))
model.add(Conv2D(64, (4,4), activation='relu'))
model.add(Conv2D(30, (4,4), padding='valid'))  
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.25))   
model.add(Conv2D(15, (4,4), activation='relu'))
model.add(Flatten())                     
model.add(Dense(120, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(100, activation='softmax'))

# 컴파일, 훈련 matrics=['acc']
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='accuracy') # 2진 분류에서는 binary_crossentropy 를 쓰면된다.



model.fit(x_train, y_train, epochs=100, batch_size=300, validation_split=0.02)

loss = model.evaluate(x_test, y_test)     
print('lose : ', loss[0])
print('accuracy : ', loss[1])