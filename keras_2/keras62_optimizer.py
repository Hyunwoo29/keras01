import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(1000, input_dim=1))
model.add(Dense(1000))
model.add(Dense(1000))
model.add(Dense(1000))
model.add(Dense(1))

#3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam, Adagrad, Adamax, Adadelta
from tensorflow.keras.optimizers import RMSprop, SGD, Nadam

# Adam
# optimizer = Adam(lr=0.1) # 러닝레이트 적용법
# loss :  3673990.0 결과물 :  [[-2814.0637]]
# optimizer = Adam(lr=0.01) # 전꺼랑 유사
# loss :  2.710537273742375e-06 결과물 :  [[10.998399]]
# optimizer = Adam(lr=0.001) # 조금더 좋아짐
# loss :  1.4858817849017214e-05 결과물 :  [[10.99312]]

# Adagriad
# optimizer = Adagrad(lr = 0.01)
# loss :  28.568944931030273 결과물 :  [[5.718534]]
# optimizer = Adagrad(lr = 0.001)
# loss :  7.379676958407799e-08 결과물 :  [[10.999941]]
# optimizer = Adagrad(lr = 0.0001)
# loss :  0.0003387552569620311 결과물 :  [[10.977026]]

# Adamax
# optimizer = Adamax(lr = 0.01)
# loss :  0.19420067965984344 결과물 :  [[10.450555]]
# optimizer = Adamax(lr = 0.001)
# loss :  3.3017145142366644e-06 결과물 :  [[10.996129]]
# optimizer = Adamax(lr = 0.0001)
# loss :  3.1919546017888933e-05 결과물 :  [[10.988614]]

# Adadelta
# optimizer = Adadelta(lr = 0.01)
# loss :  0.0013897076714783907 결과물 :  [[11.065727]]
# optimizer = Adadelta(lr = 0.001)
# loss :  0.00023414292081724852 결과물 :  [[10.9877405]]
# optimizer = Adadelta(lr = 0.0001)
# loss :  13.168617248535156 결과물 :  [[4.5557547]]

# RMSprop
# optimizer = RMSprop(lr = 0.01)
# loss :  12628.1455078125 결과물 :  [[-169.66742]]
# optimizer = RMSprop(lr = 0.001)
# loss :  11.568727493286133 결과물 :  [[17.29838]]
# optimizer = RMSprop(lr = 0.0001)
# loss :  0.00032401917269453406 결과물 :  [[10.9629965]]

# SGD
# optimizer = SGD(lr = 0.01)
# loss :  nan 결과물 :  [[nan]]
# optimizer = SGD(lr = 0.001)
# loss :  1.5179435649770312e-05 결과물 :  [[11.006393]]
# optimizer = SGD(lr = 0.0001)
# loss :  0.0005676247528754175 결과물 :  [[10.96692]]

# Nadam
# optimizer = Nadam(lr = 0.01)
# loss :  4.4413786781660747e-07 결과물 :  [[10.99734]]
# optimizer = Nadam(lr = 0.001)
# loss :  0.09949514269828796 결과물 :  [[10.428295]]
optimizer = Nadam(lr = 0.0001)
# loss :  7.086685400281567e-06 결과물 :  [[11.005924]]

model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])
model.fit(x, y, epochs= 100, batch_size=1)

loss, mse = model.evaluate(x, y, batch_size=1)
y_pred = model.predict([11])

print('loss : ', loss, '결과물 : ', y_pred)
# loss :  5.900990345253376e-07 결과물 :  [[10.999795]]