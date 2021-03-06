#실습
# Conv1D 사용

# 실습
# Enbedding, Dense, Flatten만 사용
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
from tensorflow.python.keras.backend import conv1d
from tensorflow.python.keras.layers.core import Flatten

docs = ['너무 재밌어요', '참 최고에요', '참 잘 만든 영화에요', '추천하고 싶은 영화입니다.',
        '한 번 더 보고 싶네요', '글세요', '별로에요', '생각보다 지루해요', '연기가 어색해요',
        '재미없어요', '너무 재미없다', '참 재밌네요', '청순이가 잘 생기긴 했어요']

# 긍정 1, 부정 0
labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,1])

token = Tokenizer()
token.fit_on_texts(docs)
# print(token.word_index)
# {'참': 1, '너무': 2, '잘': 3, '재밌어요': 4, '최고에요': 5, '만든': 6, '영화에요': 7, '추천하
# 고': 8, '싶은': 9, '영화입니다': 10, '한': 11, '번': 12, '더': 13, '보고': 14, '싶네요': 15, 
# '글세요': 16, '별로에요': 17, '생각보다': 18, '지루해요': 19, '연기가': 20, '어색해요': 21, 
# '재미없어요': 22, '재미없다': 23, '재밌네요': 24, '청순이가': 25, '생기긴': 26, '했어요': 27}

#어절순 으로 자름
x = token.texts_to_sequences(docs)
# print(x) # [[2, 4], [1, 5], [1, 3, 6, 7], [8, 9, 10], [11, 12, 13, 14, 15], [16], [17], [18, 19], [20, 21], [22], [2, 23], [1, 24], [25, 3, 26, 27]]

from tensorflow.keras.preprocessing.sequence import pad_sequences
# pad_x = pad_sequences(x, padding='pre', maxlen=5) #padding -> 0을 앞에 넣을거면  'pre'
# print(pad_x)
# print(pad_x.shape)  # (13, 5)
# [[ 0  0  0  2  4]
#  [ 0  0  0  1  5]
#  [ 0  1  3  6  7]
#  [ 0  0  8  9 10]
#  [11 12 13 14 15]
#  [ 0  0  0  0 16]
#  [ 0  0  0  0 17]
#  [ 0  0  0 18 19]
#  [ 0  0  0 20 21]
#  [ 0  0  0  0 22]
#  [ 0  0  0  2 23]
#  [ 0  0  0  1 24]
#  [ 0 25  3 26 27]]  ---> 이것을 원핫 인코딩 해야함.
# 0을 앞에다가 채워주는 이유는 0이 뒤에 있으면  LSTM처리할때 값이 저장되어 계속 순차적 처리되어 마지막 0값을 뽑아내가
# 때문에 원래 값 맨 뒤에값을 뽑아내기위해서 0을 앞으로 보냄

pad_x = pad_sequences(x, padding='post', maxlen=5)  # --> post쓰면 0이 뒤로감 
print(pad_x)
print(pad_x.shape) # (13, 5)  --> 원핫인코딩하면 (13, 5) -> (13, 5, 27)
# [[ 2  4  0  0  0]
#  [ 1  5  0  0  0]
#  [ 1  3  6  7  0]
#  [ 8  9 10  0  0]
#  [11 12 13 14 15]
#  [16  0  0  0  0]
#  [17  0  0  0  0]
#  [18 19  0  0  0]
#  [20 21  0  0  0]
#  [22  0  0  0  0]
#  [ 2 23  0  0  0]
#  [ 1 24  0  0  0]
#  [25  3 26 27  0]]
word_size = len(token.word_index)
print(word_size)
print(np.unique(pad_x))

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Embedding, LSTM, Input, Conv1D


# 실습, 함수형으로 고쳐봐라
input1 = Input(shape=(5,)) # Input(shape=(None,)) 이렇게 넣어줘도 가능하다
dense1 = Embedding(input_dim=28, output_dim=77)(input1)
dense2 = Conv1D(23, 2)(dense1)
dense3 = Dense(10, activation='relu')(dense2)
dense4 = Flatten()(dense3)
output1 = Dense(2)(dense4)

model = Model(inputs=input1, outputs=output1)


model.summary()
# Layer (type)                 Output Shape              Param #
# =================================================================
# input_1 (InputLayer)         [(None, 5)]               0
# _________________________________________________________________
# embedding (Embedding)        (None, 5, 77)             2156
# _________________________________________________________________
# lstm (LSTM)                  (None, 4)                 1312
# _________________________________________________________________
# dense (Dense)                (None, 10)                50
# _________________________________________________________________
# dense_1 (Dense)              (None, 2)                 22
# =================================================================

#컴파일 , 훈련
import time
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
start_time = time.time()
model.fit(pad_x, labels, epochs=100, batch_size=16)
end_time = time.time() - start_time
# 평가 , 예측
acc = model.evaluate(pad_x,labels)[1]
print("acc: ", acc) 
print('걸린시간 : ', end_time)