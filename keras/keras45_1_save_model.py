import numpy as np
from tensorflow.keras.datasets import cifar100, mnist
from icecream import ic

# 확장자는  .h5

'''
# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()
ic(x_train.shape, y_train.shape)   
ic(x_test.shape, y_test.shape)     
# 1-2. x 데이터 전처리 - scaler:2차원에서만 가능
x_train = x_train.reshape(60000, 28 * 28)   # 4차원 -> 2차원
x_test = x_test.reshape(10000, 28 * 28)
# 전처리 하기 -> scailing
print(x_train.shape, x_test.shape) # (50000, 3072)-2차원, (10000, 3072)-2차원
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, PowerTransformer, QuantileTransformer
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)   # fit_transform 은  x_train 에서만 사용한다!!!!!!!!!!!!
x_test = scaler.transform(x_test)
x_train = x_train.reshape(60000, 28, 28, 1)    # 4차원으로 다시 reshape(Conv2d 사용해야 되니까)
x_test = x_test.reshape(10000, 28, 28, 1)
# 1-3. y 데이터 전처리 -> one-hot-encoding
ic(np.unique(y_train))    # 100개
# from tensorflow.keras.utils import to_categorical   # 0,1,2 값이 없어도 무조건 생성/shape유연
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)
# print(y_train.shape, y_test.shape)
from sklearn.preprocessing import OneHotEncoder    # sklearn으로 되어 있는 애들은 모두 2차원으로 해줘야 함/OneHotEncoder는 무조건 2차원으로 해줘야 함
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
one = OneHotEncoder()
# one.fit(y_train)
y_train = one.fit_transform(y_train).toarray()   # (50000, 100)
y_test = one.transform(y_test).toarray()     # (10000, 100)
# ic(y_train.shape, y_test.shape)
'''

# 2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout, GlobalAveragePooling2D
model = Sequential()
model.add(Conv2D(128, kernel_size=(2, 2), 
                    padding='valid', input_shape=(28, 28, 1), activation='relu'))
# model.add(Dropout(0, 2)) # 20% node Dropout
model.add(Dropout(0.2))
model.add(Conv2D(128, (2,2), padding='same', activation='relu'))   
model.add(MaxPool2D()) 
model.add(Conv2D(128, (2,2),padding='valid', activation='relu'))  
model.add(Dropout(0.2))
model.add(Conv2D(128, (2,2), padding='same', activation='relu')) 
model.add(MaxPool2D()) 
model.add(Conv2D(64, (2,2), padding='valid', activation='relu')) 
model.add(Dropout(0.2))
model.add(Conv2D(64, (2,2), padding='same', activation='relu')) 
model.add(MaxPool2D()) 
model.add(Flatten()) 
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()

model.save('./_save/keras45_1_save_model.h5')  # 모델 저장        # 저장되는 확장자 : h5         # ./ : 현재위치(STUDY 폴더)