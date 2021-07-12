from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
import time

#1. 데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10], 
                [1, 1.1, 1.2, 1.3 ,1.4, 1.5, 1.6, 1.5, 1.4, 1.3],
                [10,9,8,7,6,5,4,3,2,1]])
print(x.shape) # (3,10)

x = np.transpose(x)
print(x.shape) # (10,3)

y = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
print(y.shape) # (10,)

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=3))
model.add(Dense(10))
model.add(Dense(25))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
start = time.time()
model.fit(x, y, epochs=100, batch_size=1, verbose=1)
end = time.time() - start
print("걸린시간 : ", end)
# verbose
# 0
# 걸린시간 :  2.137528657913208
# 1
# 걸린시간 :  2.8105340003967285
# 2
# 걸린시간 :  2.3855302333831787
# 3
# 걸린시간 :  2.319563627243042




#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss :', loss)


x_pred = model.predict(np.array([[10, 1.3, 1]]))
print(x_pred) 
print(x.shape)

# 1. mae란 지표를 찾을것
# 2. rmse인 지표를 찾을것