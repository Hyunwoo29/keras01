from tensorflow.keras.applications import VGG19, EfficientNetB0
from tensorflow.keras.layers import Dense, Flatten,BatchNormalization, Activation, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from keras.datasets import cifar10,cifar100
from keras.applications.vgg19 import preprocess_input
from keras.optimizers import Adam
import numpy as np
vgg16 = EfficientNetB0(weights='imagenet', include_top=False,input_shape=(32,32,3))
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
# 걸린시간 : 84.21846652030945
# loss :  1.0501272916793823
# acc : 0.5846999788284302

# trainable True
# 걸린시간 : 81.36472034454346
# loss :  1.045633635520935
# acc :  0.8104000282287598

# cifar100
# trainable False
# 걸린시간 :  141.92054200172424
# loss :  2.475953483581543
# acc : 0.34129999470710754

# trainable True
# 걸린시간 :  91.21405109405518
# loss :  2.4187075519561768
# acc :  0.5522210288963318