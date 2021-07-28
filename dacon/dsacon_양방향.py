import numpy as np
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score,log_loss
from sklearn.model_selection import StratifiedKFold

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Dropout, Bidirectional
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import plot_model, to_categorical

import warnings 
warnings.filterwarnings(action='ignore')

PATH = './_data/'
train = pd.read_csv(PATH + "train_data.csv")
test = pd.read_csv(PATH + "test_data.csv")
submission = pd.read_csv(PATH + "sample_submission.csv")

# train.groupby(train.topic_idx).size().reset_index(name="counts").plot.bar(x='topic_idx',title="Samples per each class (Training set)")
# plt.show()

train['doc_len'] = train.title.apply(lambda words: len(words.split()))

# def plot_doc_lengths(dataframe):
#     mean_seq_len = np.round(dataframe.doc_len.mean()).astype(int)
#     sns.distplot(tuple(dataframe.doc_len), hist=True, kde=True, label='Document lengths')
#     plt.axvline(x=mean_seq_len, color='k', linestyle='--', label=f'Sequence length mean:{mean_seq_len}')
#     plt.title('Document lengths')
#     plt.legend()
#     plt.show()
#     print(f" 가장 긴 문장은 {train['doc_len'].max()} 개의 단어를, 가장 짧은 문장은 {train['doc_len'].min()} 개의 단어를 가지고 있습니다.")

# plot_doc_lengths(train)
# 가장 긴 문장은 13 개의 단어를, 가장 짧은 문장은 1 개의 단어를 가지고 있습니다.

X_train = np.array([x for x in train['title']])
X_test = np.array([x for x in test['title']])
Y_train = np.array([x for x in train['topic_idx']])

# print(X_train.shape) # (45654,)
# print(X_test.shape) # (9131,)
# print(Y_train.shape) # (45654,)

from tensorflow.keras.preprocessing.text import Tokenizer
vocab_size = 2000  
tokenizer = Tokenizer(num_words = vocab_size)  
tokenizer.fit_on_texts(X_train)
sequences_train = tokenizer.texts_to_sequences(X_train)
sequences_test = tokenizer.texts_to_sequences(X_test)
# print(len(sequences_train), len(sequences_test)) # 45654 9131

word_index = tokenizer.word_index

max_length = 14
padding_type='post'
train_x = pad_sequences(sequences_train, padding='post', maxlen=max_length)
test_x = pad_sequences(sequences_test, padding=padding_type, maxlen=max_length)
# print(train_x.shape, test_x.shape) # (45654, 14) (9131, 14)

# 데이터 전처리
train_y = to_categorical(Y_train) # Y_train 원핫 인코딩
# print(train_y)
# print(train_y.shape)
# [[0. 0. 0. ... 1. 0. 0.]
#  [0. 0. 0. ... 1. 0. 0.]
#  [0. 0. 0. ... 1. 0. 0.]
#  ...
#  [0. 1. 0. ... 0. 0. 0.]
#  [0. 0. 1. ... 0. 0. 0.]
#  [0. 0. 1. ... 0. 0. 0.]]
# (45654, 7)

#파라미터 설정
vocab_size = 2000 # 제일 많이 사용하는 사이즈
embedding_dim = 200  
max_length = 14    # 위에서 그래프 확인 후 정함
padding_type='post'

model3 = Sequential([Embedding(vocab_size, embedding_dim, input_length =max_length),
        tf.keras.layers.Bidirectional(LSTM(units = 64, return_sequences = True)),
        tf.keras.layers.Bidirectional(LSTM(units = 64, return_sequences = True)),
        tf.keras.layers.Bidirectional(LSTM(units = 64)),
        Dense(7, activation='softmax')    # 결과값이 0~4 이므로 Dense(5)
    ])
    
model3.compile(loss= 'categorical_crossentropy', #여러개 정답 중 하나 맞추는 문제이므로 손실 함수는 categorical_crossentropy
              optimizer= 'adam',
              metrics = ['accuracy']) 
# model3.summary()


history = model3.fit(train_x, train_y, epochs=10, batch_size=100, validation_split= 0.2)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.title('loss of Bidirectional LSTM (model3) ', fontsize= 15)
plt.plot(history.history['loss'], 'b-', label='loss')
plt.plot(history.history['val_loss'],'r--', label='val_loss')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.title('accuracy of Bidirectional LSTM (model3) ', fontsize= 15)
plt.plot(history.history['accuracy'], 'g-', label='accuracy')
plt.plot(history.history['val_accuracy'],'k--', label='val_accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()