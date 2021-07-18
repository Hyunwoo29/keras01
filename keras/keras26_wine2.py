import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow.python.keras.utils.np_utils import to_categorical
# ./ -> 현재폴더  ../ -> 상위폴더
datasets = pd.read_csv('./_data/winequality-white.csv', sep=';', index_col=False, header=0)

# print(datasets.isnull().sum()) # 결측치 확인
# datasets = datasets.to_numpy()

x = datasets[:,0:11]
y = datasets[:,[-1]]

# 원핫 인코딩
# y = to_categorical(y)
# print(y.shape) # (4898, 10) # 0 1 2 자동채움 -> label : 10
# print(np.unique(y)) # [3. 4. 5. 6. 7. 8. 9.]
onehot_encoder = OneHotEncoder() # 0 1 2 자동채움X
onehot_encoder.fit(y)
y = onehot_encoder.transform(y).toarray() 

# print(y.shape) # (4898, 7)
# print(datasets)
# print(datasets.shape) # (4898, 12)
# print(datasets.info())
# print(datasets.describe())

# 다중분류
# 모델링하고
# 0.8 이상 완성!
# 1. 판다스 -> 넘파이
# 2. x와  y를 분리
# 3. sklearn의 onhot 사용할것
# 4. y의 라벨을 확인 np. unique(y)



# x = datasets.data
# y = datasets.target

# from tensorflow.keras.utils import to_categorical
# y = to_categorical(y)
# # print(y.shape) #(4898, 10)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=66)

from sklearn.preprocessing import StandardScaler, MaxAbsScaler, QuantileTransformer
# scaler = QuantileTransformer()
# scaler = MaxAbsScaler()
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test) 

model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(11,)))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(7, activation='softmax')) # 다중분류


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='accuracy') # 2진 분류에서는 binary_crossentropy 를 쓰면된다.

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', patience=3, mode='min', verbose=1)

model.fit(x_train, y_train, epochs=100, batch_size=2, validation_split=0.2, callbacks=[es])

loss = model.evaluate(x_test, y_test)     
print('lose : ', loss[0])
print('accuracy : ', loss[1])
# lose :  2.300898313522339
# accuracy :  0.6122449040412903
# print("=========== 예측 =============")
# print(y_test)
# y_predict = model.predict(x_test)
# print(y_predict)
