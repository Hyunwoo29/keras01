# overfit을 극복하자!!
#1. 전체 훈련 데이터가 마니 마니!!!
#2. normalization
#3. dropout

#1. 전처리

from tensorflow.keras.datasets import cifar100

import numpy as np
from tensorflow.python.keras.layers.core import Dropout

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

x_train = x_train.reshape(50000, 32 * 32 * 3)
x_test = x_test.reshape(10000, 32 * 32 * 3)

# x_train = x_train/255.
# x_test = x_test/ 255.

#스케일러
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, PowerTransformer
scaler = MinMaxScaler()
# x_train = scaler.fit(x_train)
# x_train = scaler.transform(x_train)
x_train = scaler.fit_transform(x_train) # fit_transform은 기준점이 달라질수 있어서 트레인에서만 한다
x_test = scaler.transform(x_test)

x_train = x_train.reshape(50000, 32, 32, 3)
x_test = x_test.reshape(10000, 32, 32, 3)

# from sklearn.preprocessing import OneHotEncoder
# one = OneHotEncoder()
# y_train = y_train.reshape(-1,1)
# y_test = y_test.reshape(-1,1)
# one.fit(y_train)
# y_train = one.transform(y_train).toarray() # (50000, 100)

# y_test = one.transform(y_test).toarray() # (10000, 100)

#카테고리컬은 0부터 시작한다.
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#2. 모델

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D

model = Sequential()
model.add(Conv2D(filters=100, kernel_size=(4,4), padding='same', input_shape=(32, 32, 3)))
model.add(Conv2D(20, (4,4), activation='relu'))
model.add(Conv2D(30, (4,4), padding='valid'))  
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.25))   
model.add(Conv2D(15, (4,4), activation='relu'))
model.add(Flatten())                     
model.add(Dense(40, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(100, activation='softmax'))

# 컴파일, 훈련 matrics=['acc']
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='accuracy') # 2진 분류에서는 binary_crossentropy 를 쓰면된다.


from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=4, mode='min', verbose=1)

import time
start_time = time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=300, validation_split=0.02, callbacks=[es])
end_time = time.time() - start_time
loss = model.evaluate(x_test, y_test) 
print('=================================================')
print("걸린시간 : ", end_time)    
print('lose : ', loss[0])
print('accuracy : ', loss[1])

# lose :  2.8996541500091553
# accuracy :  0.3188999891281128
# lose :  2.6582791805267334
# accuracy :  0.35440000891685486



import matplotlib.pyplot as plt
plt.figure(figsize=(9,5)) # 판을 깔다 라는뜻

#1
plt.subplot(2,1,1)  # 두번째꺼중에 첫번째꺼를 그려라 라는뜻
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')

#2
plt.subplot(2,1,2)  # 두번째꺼중에 첫번째꺼를 그려라 라는뜻
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_loss'])
plt.grid()
plt.title('acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend('acc', 'val_acc')

plt.show()