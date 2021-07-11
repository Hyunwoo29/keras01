from operator import index
from keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from tensorflow.python.keras import activations

x = np.array([1,2,3,4,5])
y = np.array([1,2,4,3,5])

optimizers = ['adam']
activations = ['elu', 'selu', 'relu']

r2score = []
opt = []
act = []
y_pred = []

model = Sequential()
for i in range(len(optimizers)) :
    for j in range(len(activations)) :
        model.add(Dense(100, input_dim = 1))
        model.add(Dense((1+i)*20))
        model.add(Dense((1+i)*10,activation=activations[j]))
        model.add(Dense((1+i)*5,activation=activations[j]))
        model.add(Dense(1))
        model.compile(loss = 'mse', optimizer=optimizers[i])
        model.fit(x,y,epochs=1000, batch_size=128)

        loss = model.evaluate(x,y)

        y_predict = model.predict(x)
        y_pred.append(y_predict)

        from sklearn.metrics import r2_score
        r2 = r2_score(y,y_predict)
        r2score.append(r2)
        opt.append(optimizers[i])
        act.append(activations[j])

index = r2score.index(max(r2score))
print("x의값 : ", y_pred[index])
print("r2score : ", r2score[index])
