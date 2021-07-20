from operator import index
from keras import optimizers
from tensorflow.keras.models import Sequential
import numpy as np
from tensorflow.python.keras import activations
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout

#1. 데이터
x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6]]) # (4, 3)
y = np.array([4,5,6,7]) # (4,)
x = x.reshape(4, 3, 1)
optimizers = ['adam']
activations = ['elu', 'selu', 'relu']


y_pred = []

model = Sequential()
for i in range(len(optimizers)) :
    for j in range(len(activations)) :
        model.add(SimpleRNN(units=10, activation='relu', input_shape=(3, 1)))
        model.add(Dense((20)))
        model.add(Dense((1+i)*10,activation=activations[j]))
        model.add(Dense((1+i)*5,activation=activations[j]))
        model.add(Dense(1))
        model.compile(loss = 'mse', optimizer=optimizers[i])
        model.fit(x,y,epochs=100, batch_size=1)
        x_input = np.array([5,6,7]).reshape(1,3,1)
        y_predict = model.predict(np.array(x_input))
        y_pred.append(y_predict)

        


# index = r2score.index(max(r2score))
results = y_pred(max(y_pred))
print(results)
# index = y_pred.index(max(y_pred))
# print("x의값 : ", y_pred[index])