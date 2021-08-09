from sklearn.preprocessing import MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer, StandardScaler, OneHotEncoder
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
from sklearn.datasets import load_iris
from icecream import ic
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Conv1D, Flatten
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import time
from sklearn.metrics import r2_score
import pandas as pd



#1. 데이터
datasets = pd.read_csv('./_data/winequality-white.csv', sep=';', index_col=None, header=0 )

# ic(datasets)

x = datasets.iloc[:,0:11]
y = datasets.iloc[:,[-1]]

# ic(x, y)
# ic(x.shape, y.shape) 

ic(np.unique(y))

y = OneHotEncoder().fit_transform(y).toarray()


# ic(y)



x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.7, random_state=60) # train 309, test 133

# scaler = QuantileTransformer()
# scaler = StandardScaler()
# scaler = PowerTransformer()
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

ic(x_train.shape, x_test.shape)






model = Sequential()
model.add(LSTM(40, activation='relu', input_shape=(11,1), return_sequences=True))
model.add(Conv1D(128, 2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Flatten())
model.add(Dense(7, activation='softmax'))

model.summary()

#3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam, Nadam
optimizers = Adam(lr = 0.01)

model.compile(loss='categorical_crossentropy', optimizer=optimizers)
es = EarlyStopping(monitor='acc', patience=20, mode='auto', verbose=1)
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

# ic| loss[0]: 1.2381054162979126
# ic| loss[1]: 0.5469387769699097
# ic| f'{걸린시간}분': '3.5분'

# Adam = 0.01
# ic| r2: 0.0848329418821453
# ic| f'{걸린시간}분': '3.5분'