from sklearn import metrics, model_selection
from sklearn.datasets import load_boston

import numpy as np
from tensorflow.python.keras.layers.core import Dropout

datasets = load_boston()
x = datasets.data
y = datasets.target
print(x.shape)
print(y.shape)
# (506, 13)  input은 13개
# (506,)   output은 1개
print(datasets.feature_names)
#['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO'
#  'B' 'LSTAT'] 'B'는 흑인의 비율
print(datasets.DESCR)

# 완료하시오!


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.7,
                                                    shuffle=True,
                                                    random_state=66)

optimizers = ['adam']
activations = ['elu', 'selu', 'relu']

r2score = []
opt = []
act = []
y_pred = []

#model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout

model = Sequential()
for i in range(len(optimizers)) :
    for j in range(len(activations)) :
        model.add(Dense(64,activation=activations[j], input_dim = 13))
        model.add(Dense(128,activation=activations[j]))
        model.add(Dense(128,activation=activations[j]))
        model.add(Dropout(0.1*i))
        model.add(Dense(128,activation=activations[j]))
        model.add(Dense(256,activation=activations[j]))
        model.add(Dense(256,activation=activations[j]))
        model.add(Dense(1))
        model.compile(loss = 'mae', optimizer=optimizers[i])
        model.fit(x_train, y_train, epochs = 1000, batch_size= 16)

        loss = model.evaluate(x_test,y_test)

        y_predict = model.predict(x_test)
        y_pred.append(y_predict)

        from sklearn.metrics import r2_score
        r2 = r2_score(y_test,y_predict)
        r2score.append(r2)
        opt.append(optimizers[i])
        act.append(activations[j])

print("loss : ", loss)
y_predict = model.predict(x_test)
index = r2score.index(max(r2score))
print("x의값 : ", y_pred[index])
print("best optimizer : ",opt[index],", best activation : ",act[index],", r2score : ", r2score[index])

# r2score :  0.8144539732699316
#  r2score :  0.8243615325835015
# r2score :  0.8307976724104797