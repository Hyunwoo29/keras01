# overfit을 극복하자!!
#1. 전체 훈련 데이터가 마니 마니!!!
#2. normalization
#3. dropout

#1. 전처리

from tensorflow.keras import optimizers
from tensorflow.keras.datasets import cifar100

import numpy as np
from tensorflow.python.keras.backend import batch_normalization
from tensorflow.python.keras.layers.core import Dropout

(x_train, y_train), (x_test, y_test) = cifar100.load_data()
x_train = x_train/255.0
x_test = x_test/ 255.0
x_train = x_train.reshape(50000, 32 * 32 * 3)
x_test = x_test.reshape(10000, 32 * 32 * 3)



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
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization

# model = Sequential()
# model.add(Conv2D(filters=100, kernel_size=(4,4), padding='same', input_shape=(32, 32, 3)))
# model.add(Conv2D(20, (4,4), padding='same', activation='relu'))
# model.add(Conv2D(30, (4,4), padding='valid'))  
# model.add(MaxPooling2D(2,2))
# model.add(Dropout(0.25))   
# model.add(Conv2D(15, (4,4), activation='relu'))                    
# model.add(Dense(40, activation='relu'))
# model.add(Dense(20, activation='relu'))
# model.add(GlobalAveragePooling2D())
# model.add(Dense(100, activation='softmax'))

model = Sequential()
model.add(Conv2D(32,kernel_size=(3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(32,32,3)))
model.add(BatchNormalization())
model.add(Conv2D(32,kernel_size=(3, 3), activation='relu', kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64,kernel_size=(3, 3), activation='relu', kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(Conv2D(64,kernel_size=(3, 3), activation='relu', kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(100, activation='softmax'))


# 컴파일, 훈련 matrics=['acc']
from tensorflow.keras.optimizers import Adam, SGD
optimizers = Adam(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizers, metrics=['acc']) # 2진 분류에서는 binary_crossentropy 를 쓰면된다.

# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='accuracy') # 2진 분류에서는 binary_crossentropy 를 쓰면된다.


from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
es = EarlyStopping(monitor='val_loss', patience=5, mode='min', verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto', verbose=1, factor=0.5)


import time
start_time = time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=124, validation_split=0.02, verbose=1, callbacks=[es, reduce_lr])
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


# adam = 0.01
# 걸린시간 :  47.17841410636902
# lose :  2.2632439136505127
# accuracy :  0.4189999997615814

# adam = 0.001
# 걸린시간 :  52.05178785324097
# lose :  2.0151288509368896
# accuracy :  0.48399999737739563

# 걸린시간 :  48.26872658729553
# lose :  1.9488461017608643
# accuracy :  0.4934999942779541

# 걸린시간 :  74.47111654281616
# lose :  1.9192038774490356
# accuracy :  0.5037000179290771


# SGD = 0.01
# 걸린시간 :  85.5053882598877
# lose :  2.773653984069824
# accuracy :  0.313400000333786
