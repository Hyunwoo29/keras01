import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow.python.keras.utils.np_utils import to_categorical
# ./ -> 현재폴더  ../ -> 상위폴더
datasets = pd.read_csv('./_data/winequality-white.csv', sep=';', index_col=False, header=0)
datasets = datasets.to_numpy()
x = datasets[:,0:11]
y = datasets[:,[-1]]

# y = to_categorical(y)
# print(y.shape) # (4898, 10) # 0 1 2 자동채움 -> label : 10
# print(np.unique(y)) # [3. 4. 5. 6. 7. 8. 9.]
onehot_encoder = OneHotEncoder() # 0 1 2 자동채움X
onehot_encoder.fit(y)
y = onehot_encoder.transform(y).toarray() 


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.7,
                                                    shuffle=True,
                                                    random_state=66)
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, QuantileTransformer
# scaler = QuantileTransformer()
# scaler = MaxAbsScaler()
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test) 
optimizers = ['adam']
activations = ['elu', 'selu', 'relu']

r2score = []
opt = []
act = []
y_pred = []

#model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.callbacks import EarlyStopping
model = Sequential()
for i in range(len(optimizers)) :
    for j in range(len(activations)) :
        model.add(Dense(128,activation=activations[j], input_shape=(11,)))
        model.add(Dense(64,activation=activations[j]))
        model.add(Dense(64,activation=activations[j]))
        model.add(Dropout(0.1*i))
        model.add(Dense(64,activation=activations[j]))
        model.add(Dense(32,activation=activations[j]))
        model.add(Dense(64,activation=activations[j]))
        model.add(Dense(7, activation='softmax'))
        model.compile(loss = 'categorical_crossentropy', optimizer=optimizers[i], metrics='accuracy')
        es = EarlyStopping(monitor='loss', patience=3, mode='min', verbose=1)
        model.fit(x_train, y_train, epochs = 100, batch_size= 6, validation_split=0.3, callbacks=[es])

        loss = model.evaluate(x_test,y_test)

        y_predict = model.predict(x_test)
        y_pred.append(y_predict)

        
        opt.append(optimizers[i])
        act.append(activations[j])
   
print('lose : ', loss[0])
print('accuracy : ', loss[1])
