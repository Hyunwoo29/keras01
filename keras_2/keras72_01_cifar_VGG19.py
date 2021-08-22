from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import Dense, Flatten,BatchNormalization, Activation, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from keras.datasets import cifar10,cifar100
from keras.applications.vgg19 import preprocess_input
from keras.optimizers import Adam
import numpy as np
vgg16 = VGG19(weights='imagenet', include_top=False,input_shape=(32,32,3))
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
history =model.fit(x_train,y_train, epochs=100, batch_size=120, validation_split=0.2,  
                                     callbacks = [early_stopping,lr])
end = time.time() - start_time

loss = model.evaluate(x_test,y_test, batch_size=1)
print("걸린시간 : ", end)
print("loss : ",loss[0])
print("acc : ",loss[1])

# cifar10
# trainable False
# 걸린시간 :  349.8994665145874
# loss :  1.0497325658798218
# acc :  0.6259999871253967
# trainable True
# 걸린시간 :  270.51390266418457
# loss :  0.6037418246269226
# acc :  0.8450000286102295

# cifar100
# trainable False
# 걸린시간 :  350.9355716705322
# loss :  2.8415517807006836
# acc :  0.3142000138759613

# trainable True
# 걸린시간 :  593.7428534030914
# loss :  1.9989005327224731
# acc :  0.5730000138282776