from sklearn import metrics, model_selection
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score
import numpy as np
from tensorflow.python.keras.layers.core import Dropout

datasets = load_boston()
x = datasets.data
y = datasets.target
print(x.shape)
print(y.shape)
# (506, 13)  input은 13개
# (506,)   output은 1개
print(datasets.feature_names)
#['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO'
#  'B' 'LSTAT'] 'B'는 흑인의 비율
print(datasets.DESCR)

# 완료하시오!


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.7,
                                                    shuffle=True,
                                                    random_state=66)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer
scaler = RobustScaler()
# scaler.fit(x_train) 
# x_train = scaler.transform(x_train) 
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1, 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1, 1)


#model
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Conv2D, Flatten, Dropout, GlobalAveragePooling2D, MaxPool2D

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=2,                          
                        padding='same', activation='relu', input_shape=(13, 1, 1))) 
model.add(Dropout(0.2))
model.add(Conv2D(32, 2, padding='same', activation='relu'))
# model.add(MaxPool2D())
model.add(Conv2D(64, 2, padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(64, 2, padding='same', activation='relu'))
# model.add(MaxPool2D())
model.add(Conv2D(128, 2, padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(128, 2, padding='same', activation='relu'))
# model.add(MaxPool2D())
model.add(GlobalAveragePooling2D()) # 평균내서 값을 뽑아주는것.
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=1)

import time 
start_time = time.time()
model.fit(x_train, y_train, epochs=1000, batch_size=32, verbose=2,
    validation_split=0.15, callbacks=[es])
end_time = time.time() - start_time

y_predict = model.predict([x_test])
loss = model.evaluate(x_test, y_test)

print("time = ", end_time)
print('loss : ', loss)
r2 = r2_score(y_test, y_predict)
print('R^2 score : ', r2)

# time =  22.726332902908325
# loss :  13.370774269104004
# R^2 score :  0.8381597446792671
