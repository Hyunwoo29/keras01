import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
#데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 전처리 해야함
x_train = x_train.reshape(60000, 28, 28, 1)/255.
x_test = x_test.reshape(10000, 28, 28, 1)/255.


# print(np.unique(y_train)) # [0 1 2 3 4 5 6 7 8 9]

# from sklearn.preprocessing import OneHotEncoder
# one = OneHotEncoder()
# y_train = y_train.reshape(-1,1)
# y_test = y_test.reshape(-1,1)
# one.fit(y_train)
# y_train = one.transform(y_train).toarray() # (60000, 10)
# y_test = one.transform(y_test).toarray() # (10000, 10)
#2. 모델

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D

model = Sequential()
model.add(Conv2D(filters=100, kernel_size=(2,2), padding='same', input_shape=(28, 28, 1)))
model.add(Conv2D(20, (4,4), activation='relu')) #(N, 9, 9, 20)
model.add(Conv2D(30, (4,4), padding='valid'))  # (N, 8, 8, 30)
model.add(MaxPooling2D(2,2))   #(N, 4, 4, 30)
model.add(Conv2D(15, (4,4)))
model.add(Flatten())     # (N, 180)                 
model.add(Dense(64, activation='relu'))
model.add(Dense(32))
model.add(Dense(10, activation='softmax'))

# 컴파일, 훈련 matrics=['acc']
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics='accuracy') # 2진 분류에서는 binary_crossentropy 를 쓰면된다.

from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
es = EarlyStopping(monitor='val_loss', patience=8, mode='min', verbose=1)
tb = TensorBoard(log_dir='./_save/_graph', histogram_freq=0,
                    write_graph=True, write_images=True)
model.fit(x_train, y_train, epochs=100, batch_size=300, validation_split=0.02, callbacks=[es,tb])

loss = model.evaluate(x_test, y_test)     
print('lose : ', loss[0])
print('accuracy : ', loss[1])

# 평가 , 예측 predict 할 필요 없다

# acc로만 판단해보자

#만들어야함!
# 0.98 이상 선착순 3명 커피

# lose :  0.04199787974357605
# accuracy :  0.9900000095367432

# lose :  0.03559916466474533
# accuracy :  0.9912999868392944