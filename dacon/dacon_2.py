from tokenize import Token
from icecream import ic
import time
import datetime
import numpy as np
from numpy import argmax
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer, OneHotEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.layers.core import Dropout, Flatten

datasets_train = pd.read_csv('./_data/train_data.csv', header=0)
datasets_test = pd.read_csv('./_data/test_data.csv', header=0)

# null값 제거
# datasets_train = datasets_train.dropna(axis=0)
# datasets_test = datasets_test.dropna(axis=0)

x = datasets_train.iloc[:, -2]
y = datasets_train.iloc[:, -1]
x_pred = datasets_test.iloc[:, -1]


token = Tokenizer()
token.fit_on_texts(x)
# ic(token.word_index)
# '4거래일째': 59159,
#                        '4건': 7359,
#                        '4경기': 2167,
#                        '4경기서': 18711,
#                        '4경기째': 29262,

# token.fit_on_texts(x_pred)

x1 = token.texts_to_sequences(x)
# ic(len(x1)) # ic| len(x1): 45654
x1_len = max(len(i) for i in x1)
# ic(x1_len) # ic| x1_len: 13
pred = token.texts_to_sequences(x_pred)
pred_len = max(len(i) for i in pred)
# ic(pred) # ic| pred_len: 11

x1_pad = pad_sequences(x1, padding='pre', maxlen = 13)
# ic(x1_pad.shape) # (45654, 13)

pred_pad = pad_sequences(pred, padding='pre', maxlen=13)
# ic(pred_pad.shape) # (9131, 13)
y = to_categorical(y)
# ic(y.shape) # (45654, 7)
# word_size = len(token.word_index)
# ic(word_size) # ic| word_size: 101081
# ic(np.unique(pred_pad)) # array([     0,      1,      2, ..., 101000, 101032, 101068])

x_train, x_test, y_train, y_test = train_test_split(x1_pad, y, train_size=0.7, shuffle=True, random_state=66)
np.save('./_save/_npy/dacon_x_train.npy', arr=x_train)
np.save('./_save/_npy/dacon_y_train.npy', arr=y_train)
np.save('./_save/_npy/dacon_x_test.npy', arr=x_test)
np.save('./_save/_npy/daconk_y_test.npy', arr=y_test)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, GRU, Bidirectional

# input = Input((13, ))
# em = Embedding(101082, 100)(input)
# x_1 =Bidirectional(LSTM(128, activation='relu', return_sequences=True))(em)
# x_2 = Dropout(0.2)(x_1)
# x_3 =Bidirectional(LSTM(64, activation='relu', return_sequences=True))(x_2)
# x_4 = Dropout(0.2)(x_3)
# x_5 =Bidirectional(LSTM(32, activation='relu', return_sequences=True))(x_4)
# x_6 = Dropout(0.2)(x_5)
# x_7 =Bidirectional(LSTM(16, activation='relu', return_sequences=True))(x_6)
# x_8 = Dropout(0.2)(x_7)
# x_9 =Bidirectional(LSTM(8, activation='relu', return_sequences=True))(x_8)
# x_10 = Flatten()(x_9)
# output = Dense(7, activation='softmax')(x_10)

# model = Model(inputs=input, outputs=output)

# input = Input((13, ))
# a = Embedding(101082, 8)(input)
# a1 = LSTM(120, activation='relu')(a)
# a2 = Dense(64, activation='relu')(a1)
# a3 = Dense(12, activation='relu')(a2)
# output = Dense(7, activation='softmax')(a3)

# model = Model(inputs=input, outputs=output)
model = Sequential()
model.add(Embedding(input_dim=101082, output_dim=77, input_length=13)) # 예시1  #length는 왠만하면 max length 길이에 맞춰줘라
# model.add(Embedding(27, 77))  # 예시2  # 인풋 27, 아웃풋 77
# model.add(Embedding(27, 77, input_length=5)) #예시3
# Embedding은  예시1과 예시2 두개중 하나로 표현 가능하다. 예시3도 가능은하다.
# model.add(LSTM(128, activation='relu', return_sequences= True))
# # model.add(LSTM(128, activation='relu', return_sequences= True))
# model.add(Dropout(0.2))
# model.add(LSTM(64, activation='relu', return_sequences= True))
# # model.add(LSTM(64, activation='relu', return_sequences= True))
# model.add(Dropout(0.2))
# model.add(LSTM(32, activation='relu', return_sequences= True))
# # model.add(LSTM(32, activation='relu', return_sequences= True))
# model.add(Dropout(0.2))
# model.add(LSTM(16, activation='relu', return_sequences= True))
# # model.add(LSTM(16, activation='relu', return_sequences= True))
# model.add(Dropout(0.2))
# model.add(LSTM(8, activation='relu', return_sequences= True))
# # model.add(LSTM(8, activation='relu', return_sequences= True))
# model.add(Flatten())
# model.add(Dense(7, activation='softmax'))



model.add(LSTM(128, activation='relu'))
model.add(Dense(64, activation= 'relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation= 'relu'))
model.add(Dense(7, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

date = datetime.datetime.now()
date_time = date.strftime('%m%d_%H%M')
path = './_save/mcp/'
info = '{epoch:02d}_{val_loss:.4f}'
filepath = ''.join([path, 'test', '_', date_time, '_', info, '.hdf5'])
es = EarlyStopping(monitor='val_loss', restore_best_weights=True, mode='min', verbose=1, patience=10)
cp = ModelCheckpoint(monitor='val_loss', save_best_only=True, mode='min', verbose=1, filepath=filepath)
start_time = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=100, verbose=1, validation_split=0.02, callbacks=[es, cp])
end_time = time.time() - start_time

#4. Evaluating
loss = model.evaluate(x_test, y_test)

ic('loss = ', loss[0])
ic('acc = ', loss[1])
ic('time taken(s) = ', end_time)


#5. Prediction
prediction = model.predict(pred_pad)
prediction = np.argmax(prediction, axis=1) # to_categorical 되돌리기
# print(type.prediction) # numpy.ndarray

# 제출파일형식 맞추기
index = np.array([range(45654, 54785)])
index = np.transpose(index)
index = index.reshape(9131, )
file = np.column_stack([index, prediction])
file = pd.DataFrame(file)
file.to_csv('./_data/sample_submission.csv', header=['index', 'topic_idx'], index=False)

# ic| 'loss = ', loss[0]: 1.7807492017745972
# ic| 'acc = ', loss[1]: 0.7303059101104736
# ic| 'time taken(s) = ', end_time: 491.0228407382965

# ic| 'loss = ', loss[0]: 1.766216516494751
# ic| 'acc = ', loss[1]: 0.7398700714111328
# ic| 'time taken(s) = ', end_time: 969.3765585422516

# ic| 'loss = ', loss[0]: 1.4456771612167358
# ic| 'acc = ', loss[1]: 0.748923122882843
# ic| 'time taken(s) = ', end_time: 908.4239401817322

# ic| 'loss = ', loss[0]: 1.1994911432266235
# ic| 'acc = ', loss[1]: 0.7506023049354553
# ic| 'time taken(s) = ', end_time: 337.4024827480316

# ic| 'loss = ', loss[0]: 0.741577684879303
# ic| 'acc = ', loss[1]: 0.7534496784210205
# ic| 'time taken(s) = ', end_time: 211.66828107833862