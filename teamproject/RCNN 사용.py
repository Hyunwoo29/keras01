from tensorflow.keras.datasets import cifar10
import numpy as np
from numpy import size

cifar10Data = tempdir;
(trainingImages, trainingLabels,testImages,testLabels) = cifar10.load_data()

size(trainingImages)





# model = Sequential()
# model.add(SimpleRNN(units=10, activation='relu', input_shape=(3, 1)))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(12, activation='relu'))
# model.add(Dense(1))

# # model.summary()
# # simple_rnn (SimpleRNN)       (None, 10)                120
# # 파라미터값이 120이 나오는지 구하라
# # ( unit 개수 * unit 개수 ) + ( input_dim(feature) 수 * unit 개수 ) + ( 1 * unit 개수)
# # (10*10) + (1*10) + (1*10) = 120
# # (Input + bias) * output + output * output --> (Input + baias + output) * output


# #3. 컴파일, 훈련
# model.compile(loss='mse', optimizer='adam')
# model.fit(x, y, epochs=100, batch_size=1)

# #4. 평가, 예측
# x_input = np.array([5,6,7]).reshape(1,3,1)
# results = model.predict(np.array(x_input))
# print(results)
