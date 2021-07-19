from PIL import Image
import os, glob, numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model,load_model,save_model
from keras.layers import Input, Activation
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization,Activation,ZeroPadding2D,Add
from keras.layers import GlobalAveragePooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
from keras import backend as K
import tensorflow as tf
from tensorflow.python.framework import ops as tf_ops
from keras.optimizers import Adam
from tensorflow.keras.datasets import cifar100


(x_train,y_train),(x_test,y_test) = cifar100.load_data()

x_train = x_train / 255.
x_test = x_test / 255.

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
one = OneHotEncoder()
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
one.fit(y_train)
y_train = one.transform(y_train).toarray()
y_test = one.transform(y_test).toarray()
print(y_train.shape)            # (50000,10)
print(y_test.shape)            # (10000,10)

x_train,x_valid, y_train, y_valid = train_test_split(x_train,y_train, train_size = 0.8, shuffle=True,random_state=66)





idg = ImageDataGenerator(
    width_shift_range=(0.1),   #
    height_shift_range=(0.1),
    )    


train_generator = idg.flow(x_train,y_train,batch_size=128)
valid_generator = idg.flow(x_valid,y_valid)

model = Sequential()
model.add(Conv2D(filters = 16, kernel_size =(1,1), padding = 'valid', strides=(1,1),
input_shape = (32,32,3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(filters = 16, kernel_size =(3,3), padding = 'same',strides=(1,1)))
model.add(BatchNormalization())
model.add(Activation('relu'))                         
model.add(Conv2D(filters = 64, kernel_size =(1,1), padding = 'valid',strides=(1,1)))
model.add(BatchNormalization())
model.add(Activation('relu')) 
model.add(MaxPooling2D((3, 3), 2))

model.add(Conv2D(filters = 32, kernel_size =(1,1), padding = 'valid',strides=(1,1)))
model.add(BatchNormalization())    
model.add(Activation('relu'))
model.add(Conv2D(filters = 32, kernel_size =(3,3), padding = 'same',strides=(1,1)))
model.add(BatchNormalization())
model.add(Activation('relu'))                              
model.add(Conv2D(filters = 128, kernel_size =(1,1), padding = 'valid',strides=(1,1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D((3, 3), 2)) 

model.add(Conv2D(filters = 64, kernel_size =(1,1), padding = 'valid', strides=(1,1)))
model.add(BatchNormalization())     
model.add(Activation('relu'))                        
model.add(Conv2D(filters = 64, kernel_size =(3,3), padding = 'same', strides=(1,1)))
model.add(BatchNormalization())
model.add(Activation('relu')) 
model.add(Conv2D(filters = 256, kernel_size =(1,1), padding = 'valid', strides=(1,1)))
model.add(BatchNormalization())
model.add(Activation('relu')) 
model.add(MaxPooling2D((3, 3), 2))
model.add(GlobalAveragePooling2D())
model.add(Dropout(0.1))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(100, activation='softmax'))
# model2 = load_model('../data/modelcheckpoint/myPproject_5.hdf5')
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001,epsilon=None), metrics=['acc'])

model.summary()
model_path = '../data/modelcheckpoint/my_project_act2.hdf5'
checkpoint = ModelCheckpoint(filepath=model_path , monitor='val_acc', verbose=1, save_best_only=True)
early_stopping = EarlyStopping(monitor='val_acc', patience=30)
lr = ReduceLROnPlateau(patience=15, factor=0.5,verbose=1)

history = model.fit_generator(train_generator,epochs=100, steps_per_epoch= len(x_train) / 128,
validation_data=valid_generator, callbacks=[early_stopping,lr,checkpoint])

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss'] 
val_loss = history.history['val_loss']


print('acc : ', acc[-1])
print('val_acc : ', val_acc[:-1])
