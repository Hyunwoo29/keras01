from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

#1. 데이터
x_train = np.array([1,2,3,4,5,6,7])  #훈련,공부는 트레인
y_train = np.array([1,2,3,4,5,6,7])
x_test = np.array([8,9,10])  #평가는 test
y_test = np.array([8,9,10])
x_val = np.array([11,12,13])
y_val = np.array([11,12,13])
#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(10))
model.add(Dense(25))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

model.fit(x_train, y_train, epochs=1000, batch_size=1, validation_data=(x_val, y_val))

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)

# result = model.predict([11])
# print('11의 예측값 :', result)
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
y_predict = model.predict([11])

# plt.scatter(x, y)
# plt.plot(x, y_predict, color='red')
# plt.show()

