import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist

(x_train, y_train) , (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape)     # (60000,28,28) (60000,)    => 60000,28,28,1
print(x_test.shape, y_test.shape)       # (10000,28,28) (10000,)    => 10000,28,28,1

print(x_train[0])   #    
print("y_train = ", y_train[0])   # 5

print(x_train[0].shape)     #(28, 28)

x_train = x_train.reshape(60000,28,28,1).astype('float32')/255.         # 이미지의 전처리 (max값이 255기때문에 255로 나눠서
                                                                                            #0~1 사이로 만듦)
x_test = x_test.reshape(10000,28,28,1)/255.
#(x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2],1) ) # x_test = x_train.reshape(10000,28,28,1)/255.

#OneHotEncoding
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
one = OneHotEncoder()
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
one.fit(y_train)
y_train = one.transform(y_train).toarray()
y_test = one.transform(y_test).toarray()
print(y_train.shape)            # (60000,10)
print(y_test.shape)            # (10000,10)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

model = Sequential()

model.add(Conv2D(filters = 100, kernel_size =(2,2), padding = 'same', strides = 1, 
                                        input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))
for a in range(4) :
    model.add(Conv2D(14-a*2,(2,2)))
    model.add(Dropout(0.2))
model.add(Flatten())
for i in range(5) :
    model.add(Dense(310+i, activation= 'relu'))
model.add(Dense(10, activation='softmax'))

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience= 5, mode = 'auto')

model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train,y_train, epochs=100, batch_size=256, validation_split=0.02,  
                                     callbacks = early_stopping)

#4. evaluate , predict

loss = model.evaluate(x_test,y_test, batch_size=12)
print("loss : ",loss[0])
print("accuracy:", loss[1])

# loss :  0.06146829202771187
# acc: 0.9873999953269958

# loss :  0.056358397006988525
# acc: 0.9861000180244446