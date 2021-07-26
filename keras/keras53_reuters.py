from numpy.lib.function_base import average
from tensorflow.keras.datasets import reuters
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

(x_train, y_train), (x_test, y_test) = reuters.load_data(
    num_words=10000, test_split=0.2
)

# print(x_train[0], type(x_train[0]))
# [1, 2, 2, 8, 43, 10, 447, 5, 25, 207, 270, 5, 3095, 111, 16, 369, 186, 90, 67, 7, 89, 5, 19, 102, 
# 6, 19, 124, 15, 90, 67, 84, 22, 482, 26, 7, 48, 4, 49, 8, 864, 39, 209, 154, 6, 151, 6, 83, 11, 15, 22, 155, 11, 15, 7, 48, 9, 4579, 
# 1005, 504, 6, 258, 6, 272, 11, 15, 22, 134, 44, 11, 15, 16, 8, 197,
# 1245, 90, 67, 52, 29, 209, 30, 32, 132, 6, 109, 15, 17, 12] <class 'list'>

print(y_train[0]) # 3
print(len(x_train[0]), len(x_train[1])) # 87 56
# print(x_train[0].shape) 
# 리스트는 쉐이프가 안찍힌다.
# 이유는, 리스트에는 int, float형 str등등 다 들어갈수있기 때문에.
# 넘파이랑, 판다스는 찍힌다. 이유는 int, float형만 들어갈수 있기때문

print(x_train.shape, x_test.shape) # (8982,) (2246,)
print(y_train.shape, y_test.shape) # (8982,) (2246,)
print(type(x_train)) # <class 'numpy.ndarray'>

print("뉴스기사의 최대길이 : ",max(len(i) for i in x_train)) # 2376
print("뉴스기사의 평균길이 : ",sum(map(len, x_train)) / len(x_train)) # 145.5398574927633

# plt.hist([len(s) for s in x_train], bins=50)
# plt.show()

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

x_train = pad_sequences(x_train, maxlen=100, padding='pre')
x_test = pad_sequences(x_train, maxlen=100, padding='pre')
print(x_train.shape, x_test.shape) # (8982, 100) (8982, 100)
print(type(x_train), type(x_train[0])) # <class 'numpy.ndarray'> <class 'numpy.ndarray'>
print(x_train[0])
#y 확인
print(np.unique(y_train))

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape, y_test.shape)

#모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding

model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=66, input_length=100))
