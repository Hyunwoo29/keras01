import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout, LSTM, Bidirectional

#1. 데이터
x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6]]) # (4, 3)
y = np.array([4,5,6,7]) # (4,)

print(x.shape, y.shape)

x = x.reshape(4, 3, 1) # (batch_size, timesteps, feature)
print(x)
#2. 모델구성
model = Sequential()
model.add(LSTM(units=10, activation='relu', input_shape=(3, 1), return_sequences=True))
model.add(Bidirectional(LSTM(units=10, activation='relu')))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(1))

#Bidirectional = LSTM * 2  이유는 양쪽을 왔다갔다 해서


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=1)

#4. 평가, 예측
x_input = np.array([5,6,7]).reshape(1,3,1)
results = model.predict(np.array(x_input))
print(results)
