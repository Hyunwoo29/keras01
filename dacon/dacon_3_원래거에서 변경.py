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
import re

train = pd.read_csv('./_data/train_data.csv', header=0)
test = pd.read_csv('./_data/test_data.csv', header=0)

# null값 제거
# datasets_train = datasets_train.dropna(axis=0)
# datasets_test = datasets_test.dropna(axis=0)

# x = datasets_train.iloc[:, -2]
# y = datasets_train.iloc[:, -1]
# x_pred = datasets_test.iloc[:, -1]
train['doc_len'] = train.title.apply(lambda words: len(words.split()))

x_train = np.array([x for x in train['title']])
x_predict = np.array([x for x in test['title']])
y_train = np.array([x for x in train['topic_idx']])

def text_cleaning(docs):
    for doc in docs:
        doc = re.sub("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "", doc)
    return docs
x = text_cleaning(x_train)
x_predict = text_cleaning(x_predict)
# ic(x.shape) ic| x.shape: (45654,)

# 불용어 제거, 특수문자 제거
# import string
# def define_stopwords(path):
#     sw = set()
#     for i in string.punctuation:
#         sw.add(i)

#     with open(path, encoding='utf-8') as f:
#         for word in f:
#             sw.add(word)

#     return sw
# x = define_stopwords(x)

from tensorflow.keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer()  
tokenizer.fit_on_texts(x)
sequences_train = tokenizer.texts_to_sequences(x)
sequences_test = tokenizer.texts_to_sequences(x_predict)

#리스트 형태의 빈값 제거
sequences_train = list(filter(None, sequences_train))
sequences_test = list(filter(None, sequences_test))

#길이 확인
# x1_len = max(len(i) for i in sequences_train)
# ic(x1_len) # ic| x1_len: 11
# x_pred = max(len(i) for i in sequences_test)
# ic(x_pred) # ic| x_pred: 9

xx = pad_sequences(sequences_train, padding='pre', maxlen = 11)
# ic(xx.shape) ic| xx.shape: (42477, 11)
yy = pad_sequences(sequences_test, padding='pre', maxlen=11)

y = to_categorical(y_train)


x_train, x_test, y_train, y_test = train_test_split(xx, y, train_size=0.7, shuffle=True, random_state=66)
np.save('./_save/_npy/dacon_x_train1.npy', arr=x_train)
np.save('./_save/_npy/dacon_y_train1.npy', arr=y_train)
np.save('./_save/_npy/dacon_x_test1.npy', arr=x_test)
np.save('./_save/_npy/dacon_y_test1.npy', arr=y_test)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, GRU, Bidirectional

model = Sequential()
model.add(Embedding(input_dim=101082, output_dim=77, input_length=11))
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
prediction = model.predict(yy)
prediction = np.argmax(prediction, axis=1) # to_categorical 되돌리기
# print(type.prediction) # numpy.ndarray

# 제출파일형식 맞추기
index = np.array([range(45654, 54785)])
index = np.transpose(index)
index = index.reshape(9131, )
file = np.column_stack([index, prediction])
file = pd.DataFrame(file)
file.to_csv('./_data/sample_submission.csv', header=['index', 'topic_idx'], index=False)

# ic| 'loss = ', loss[0]: 0.7068963646888733
# ic| 'acc = ', loss[1]: 0.7655690908432007
# ic| 'time taken(s) = ', end_time: 190.59102249145508