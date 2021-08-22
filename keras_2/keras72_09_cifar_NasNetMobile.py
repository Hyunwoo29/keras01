from tensorflow.keras.applications import VGG19, NASNetMobile
from tensorflow.keras.layers import Dense, Flatten,BatchNormalization, Activation, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from keras.datasets import cifar10,cifar100
from keras.applications.vgg19 import preprocess_input
from keras.optimizers import Adam
import numpy as np
vgg16 = NASNetMobile(weights='imagenet', include_top=False,input_shape=(32,32,3))
vgg16.trainable = False
(x_train, y_train), (x_test,y_test)= cifar100.load_data()

# 데이터 전처리

x_train = preprocess_input(x_train)
x_test = preprocess_input(x_test)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout,LSTM
model = Sequential()
model.add(vgg16)
# model.add(Conv2D(filters=1024,kernel_size=(1,1),padding='valid')) # 미세조정(파인튜닝)
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(GlobalAveragePooling2D())
model.add(Flatten())
model.add(Dense(128))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(64))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(100, activation= 'softmax'))
model.summary()


from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
early_stopping = EarlyStopping(monitor='val_loss', patience= 20)
lr = ReduceLROnPlateau(factor=0.01,verbose=1,patience=10)

model.compile(loss = 'categorical_crossentropy', optimizer=Adam(1e-5), metrics=['acc'])
import time 
start_time = time.time()
history =model.fit(x_train,y_train, epochs=100, batch_size=1024, validation_split=0.2,  
                                     callbacks = [early_stopping,lr])
end = time.time() - start_time

loss = model.evaluate(x_test,y_test, batch_size=1)
print("걸린시간 : ", end)
print("loss : ",loss[0])
print("acc : ",loss[1])

# cifar10
# trainable False
# 걸린시간 : 1820.2131909482001
# loss :  7.129i94892819182
# acc : 0.08271829281910189

# trainable True
# 걸린시간 : 2200.4734745025635
# loss :  7.454424858093262
# acc :  0.09189999979734421

# cifar100
# trainable False
# 걸린시간 :  2820.2131909482001
# loss :  12.213281211853027
# acc : 0.0082231001811981

# trainable True
# 걸린시간 :  3120.123194820012
# loss :  11.90281918281919
# acc :  0.0092000018119812