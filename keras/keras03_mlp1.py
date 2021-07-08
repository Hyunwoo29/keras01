from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

#1. 데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10], 
                [1, 1.1, 1.2, 1.3 ,1.4, 1.5, 1.6, 1.5, 1.4, 1.3]])
print(x.shape) # (2,10)

x = np.transpose(x)
print(x.shape) # (10,2)

y = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
print(y.shape) # (10,)

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=2))
model.add(Dense(10))
model.add(Dense(25))
model.add(Dense(10))
model.add(Dense(1))
#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=1)
#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss :', loss)


# x_pred = model.predict(np.array([[10, 1.3]]))
# print(x_pred) 

# 그래프
y_predict = model.predict(x)
plt.scatter(x[:,0], y)
plt.scatter(x[:,1], y)
plt.plot(x, y_predict, color='red')
plt.show()
