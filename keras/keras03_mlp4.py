from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

#1. 데이터
x = np.array([range(10)])
x = np.transpose(x)

y = np.array([[1,2,3,4,5,6,7,8,9,10], 
                [1, 1.1, 1.2, 1.3 ,1.4, 1.5, 1.6, 1.5, 1.4, 1.3],
                [10,9,8,7,6,5,4,3,2,1]])
y = np.transpose(y)


#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(10))
model.add(Dense(25))
model.add(Dense(10))
model.add(Dense(3))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=1000, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss :', loss)


x_pred = model.predict(np.array([[9]]))
print(x_pred) 

y_predict = model.predict(x)
plt.scatter(x, y[:,0])
plt.scatter(x, y[:,1])
plt.scatter(x, y[:,2])
plt.plot(x, y_predict)
plt.show()