from tensorflow.keras.datasets import imdb
import numpy as np
import pandas as pd
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import re
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, Embedding
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)

# print(len(x_test))  #x_train, y_train = 25000개
# a = max(y_train) + 1  # 카테고리 2개

# print(x_train.shape, x_test.shape) # (25000,) (25000,)
# unique_elements, counts_elements = np.unique(y_train, return_counts=True)
# print("각 레이블에 대한 빈도수:")
# print(np.asarray((unique_elements, counts_elements)))
# # [[    0     1]
# #  [12500 12500]]
x_train = pad_sequences(x_train, maxlen=500, padding='pre')
x_test = pad_sequences(x_train, maxlen=500, padding='pre')
# x_train = pad_sequences(x_train, maxlen= 500)
# x_test = pad_sequences(x_test, maxlen=500)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
# print(y_train.shape, y_test.shape) # (25000, 2) (25000, 2)
model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=77, input_length=500))
model.add(GRU(128))
model.add(Dense(2, activation='sigmoid'))

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(x_train, y_train, epochs=15, callbacks=[es], batch_size=60, validation_split=0.2)

loss = model.evaluate(x_test, y_test)

print('loss = ', loss[0])
print('acc = ', loss[1])