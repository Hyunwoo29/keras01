from sklearn.preprocessing import MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
from sklearn.datasets import load_breast_cancer
from icecream import ic
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Conv1D, Flatten
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import time
from sklearn.metrics import r2_score



#1. 데이터
datasets = load_breast_cancer()

x = datasets.data
y = datasets.target

ic(x, y) # ic| x.shape: (569, 30), y.shape: (569,)



x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.7, random_state=60) # train 309, test 133

# scaler = QuantileTransformer()
scaler = StandardScaler()
# scaler = PowerTransformer()
# scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

# ic(x_train.shape, x_test.shape)


# ic(np.unique(y))


model = Sequential()
model.add(LSTM(40, activation='relu', input_shape=(30,1), return_sequences=True))
model.add(Conv1D(128, 2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Flatten())
model.add(Dense(1,activation='sigmoid'))

# model.summary()

#3. 컴파일, 훈련

from tensorflow.keras.optimizers import Adam, Nadam
optimizers = Nadam(lr = 0.001)

model.compile(loss='binary_crossentropy', optimizer=optimizers)
es = EarlyStopping(monitor='acc', patience=5, mode='auto', verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, mode='auto', factor=0.5)

start = time.time()
model.fit(x_train, y_train, epochs=100, verbose=1, validation_split=0.2, batch_size=32, shuffle=True, callbacks=[es, reduce_lr])
걸린시간 = round((time.time() - start) /60,1)

#4. 평가, 예측

y_predict = model.predict(x_test)
loss = model.evaluate(x_test, y_test)
# ic(loss[0])
# ic(loss[1])
r2 = r2_score(y_test, y_predict)
ic(r2)
ic(f'{걸린시간}분')

# ic| loss[0]: 0.028651483356952667
# ic| loss[1]: 0.988304078578949
# ic| f'{걸린시간}분': '0.8분'


# Nadam = 0.001
# ic| r2: 0.9599138345024782
# ic| f'{걸린시간}분': '1.1분'