from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import random
# 데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,4,3,5])


# print(x_test)
# print(y_test)

# 모델구성
model = Sequential()
model.add(Dense(1, input_dim=1))
model.add(Dense(10))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(1))

# 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=1050, batch_size=1)

# 평가,예측
loss = model.evaluate(x, y)
print('loss : ', loss)

y_predict = model.predict(x)
print('x의 예측값 :', y_predict)

# 완성한뒤, 출력결과 스샷

from sklearn.metrics import r2_score
r2 = r2_score(y, y_predict)

print("r2스코어 : ", r2)

# 과제 2
# R2를 심수안씨를 이겨라!!
# 일 밤 12시까지