from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
import random

#1. 데이터
x = np.array(range(100)) 
y = np.array(range(1, 101))

x_train = np.split(x, (0, 70)) # x[:70]  
y_train = np.split(y, (0, 70)) # y[:70]  
x_test =  x_train[1]          # x[-30:]  
y_test =  y_train[2]          # y[70:]  

# print(x_train.shape, y_train.shape) #(70,) (70,)
# print(x_test.shape, y_test.shape)   #(30,) (30,)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                    train_size=0.7, shuffle=True, random_state=66)
print(x_test)
print(y_test)


# 한번에 x,y 배열 섞는법
# s = np.arange(x.shape[0])  
# np.random.shuffle(s)
# x = x[s]
# y = y[s]
# print(x)
# print(y)

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(10))
model.add(Dense(25))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

model.fit(x_train, y_train, epochs=100, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)

y_predict = model.predict(x_test)
print('100의 예측값 :', y_predict)
'''
Epoch 1000000/1000000
5/5 [==============================] - 0s 510us/step - loss: 0.4825
1/1 [==============================] - 0s 78ms/step - loss: 0.3800
loss : 0.3800259232521057
6의 예측값 : [[5.703544]]
'''
'''
10/10 [==============================] - 0s 555us/step - loss: 6.4936
1/1 [==============================] - 0s 75ms/step - loss: 3.7590
loss : 3.759042263031006
11의 예측값 : [[11.224196]]
PS D:\study> 
'''
# y_predict = model.predict([11])

# plt.scatter(x, y)
# plt.plot(x, y_predict, color='red')
# plt.show()

from sklearn.metrics import r2_score, mean_squared_error
r2 = r2_score(y_test, y_predict)

print("r2스코어 : ", r2)

def RMSE(y_test, y_predict) :
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
print("rmse스코어 : ", rmse)

# r2스코어 :  0.9999999917355956
# rmse스코어 :  0.0026881682074736584
