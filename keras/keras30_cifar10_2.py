from tensorflow.keras.datasets import cifar10

import numpy as np

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.reshape(50000, 32, 32, 3)/255.
x_test = x_test.reshape(10000, 32, 32, 3)/255.


# print(np.unique(y_train)) # [0 1 2 3 4 5 6 7 8 9]

from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder()
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
one.fit(y_train)
y_train = one.transform(y_train).toarray() # (50000, 10)

y_test = one.transform(y_test).toarray() # (10000, 10)
#2. 모델

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D

model = Sequential()
model.add(Conv2D(filters=100, kernel_size=(2,2), padding='same', input_shape=(32, 32, 3)))
model.add(Conv2D(20, (4,4), activation='relu')) 
model.add(Conv2D(30, (4,4), padding='valid'))  
model.add(MaxPooling2D())   
model.add(Conv2D(15, (4,4)))
model.add(Flatten())                     
model.add(Dense(64, activation='relu'))
model.add(Dense(32))
model.add(Dense(10, activation='softmax'))

# 컴파일, 훈련 matrics=['acc']
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='accuracy') # 2진 분류에서는 binary_crossentropy 를 쓰면된다.

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=8, mode='min', verbose=1)

model.fit(x_train, y_train, epochs=100, batch_size=300, validation_split=0.02, callbacks=[es])

loss = model.evaluate(x_test, y_test)     
print('lose : ', loss[0])
print('accuracy : ', loss[1])

# lose :  1.247612714767456
# accuracy :  0.6532999873161316
# image 32, 32, 3