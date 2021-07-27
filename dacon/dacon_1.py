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

datasets_train = pd.read_csv('./_data/train_data.csv', header=0)
datasets_test = pd.read_csv('./_data/test_data.csv', header=0)

# x, y, x_pred 분류
x = datasets_train.iloc[:, -2]
y = datasets_train.iloc[:, -1]
x_pred = datasets_test.iloc[:, -1]
# print(x.head(), y.head(), x_pred.head())

# x, x_pred 토큰화 및 sequence화
token = Tokenizer()
token.fit_on_texts(x)
x = token.texts_to_sequences(x)
x_pred = token.texts_to_sequences(x_pred)

max_len1 = max(len(i) for i in x)
avg_len1 = sum(map(len, x)) / len(x)
max_len2 = max(len(i) for i in x_pred)
avg_len2 = sum(map(len, x_pred)) / len(x_pred)

x = pad_sequences(x, padding='pre', maxlen=100)
x_pred = pad_sequences(x_pred, padding='pre', maxlen=100)


y = to_categorical(y)

# x, y train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=66)

#2. Modeling
input = Input((100, ))
a = Embedding(101082, 8)(input)
a1 = LSTM(120, activation='relu')(a)
a2 = Dense(64, activation='relu')(a1)
a3 = Dense(12, activation='relu')(a2)
output = Dense(7, activation='softmax')(a3)

model = Model(inputs=input, outputs=output)

#3. Compiling, Training
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

date = datetime.datetime.now()
date_time = date.strftime('%m%d_%H%M')
path = './_save/mcp/'
info = '{epoch:02d}_{val_loss:.4f}'
filepath = ''.join([path, 'test', '_', date_time, '_', info, '.hdf5'])
es = EarlyStopping(monitor='val_loss', restore_best_weights=False, mode='auto', verbose=1, patience=8)
cp = ModelCheckpoint(monitor='val_loss', save_best_only=True, mode='auto', verbose=1, filepath=filepath)
start_time = time.time()
model.fit(x_train, y_train, epochs=10, batch_size=100, verbose=1, validation_split=0.2, callbacks=[es, cp])
end_time = time.time() - start_time

#4. Evaluating
loss = model.evaluate(x_test, y_test)

print('loss = ', loss[0])
print('acc = ', loss[1])
print('time taken(s) = ', end_time)


#5. Prediction
prediction = model.predict(x_pred)
prediction = np.argmax(prediction, axis=1) # to_categorical 되돌리기
# print(type.prediction) # numpy.ndarray

# 제출파일형식 맞추기
index = np.array([range(45654, 54785)])
index = np.transpose(index)
index = index.reshape(9131, )
file = np.column_stack([index, prediction])
file = pd.DataFrame(file)
file.to_csv('./_data/sample_submission.csv', header=['index', 'topic_idx'], index=False)

# loss =  1.9433084726333618
# acc =  0.6229328513145447
# time taken(s) =  275.61812376976013