from tensorflow.keras.applications import VGG19, Xception
from tensorflow.keras.layers import Dense, Flatten,BatchNormalization, Activation, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from keras.datasets import cifar10,cifar100
from keras.applications.vgg19 import preprocess_input
from keras.optimizers import Adam
import numpy as np
vgg16 = Xception(weights='imagenet', include_top=False,input_shape=(96,96,3))
vgg16.trainable = False
(x_train, y_train), (x_test,y_test)= cifar10.load_data()

# 데이터 전처리

x_train = preprocess_input(x_train)
x_test = preprocess_input(x_test)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout,LSTM,UpSampling2D
model = Sequential()
model.add(UpSampling2D(size=(3,3), input_shape=(32,32,3)))
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
model.add(Dense(10, activation= 'softmax'))
# model.summary()


from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
early_stopping = EarlyStopping(monitor='val_loss', patience= 5)
lr = ReduceLROnPlateau(factor=0.01,verbose=1,patience=5)

model.compile(loss = 'categorical_crossentropy', optimizer=Adam(1e-5), metrics=['acc'])
import time 
start_time = time.time()
history =model.fit(x_train,y_train, epochs=10, batch_size=312, validation_split=0.2,  
                                     callbacks = [early_stopping,lr])
end = time.time() - start_time

loss = model.evaluate(x_test,y_test, batch_size=1)
print("걸린시간 : ", end)
print("loss : ",loss[0])
print("acc : ",loss[1])

# cifar10
# trainable False
# 걸린시간 :  110.55665993690491
# loss :  1.9717763662338257
# acc :  0.32359999418258667

# trainable True
# 걸린시간 :  442.08923745155334
# loss :  0.6347699165344238
# acc :  0.8184999823570251

# cifar100
# trainable False
# 걸린시간 :  874.7901332378387
# loss :  3.956064462661743
# acc :  0.13359999656677246

# trainable True
# 걸린시간 :  444.75471782684326
# loss :  3.8019487857818604
# acc :  0.22220000624656677