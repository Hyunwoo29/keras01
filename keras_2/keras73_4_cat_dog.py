# 실습
# 카테고리컬 시그모이드 쓸것
from tensorflow.keras.applications import VGG19, EfficientNetB0
from tensorflow.keras.layers import Dense, Flatten,BatchNormalization, Activation, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from keras.datasets import cifar10,cifar100
from keras.applications.vgg19 import preprocess_input
from keras.optimizers import Adam
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     horizontal_flip=True,
#     vertical_flip=True,
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     rotation_range=5,
#     zoom_range=1.2,
#     shear_range=7,
#     fill_mode='nearest',
#     validation_split=.2
# )
# test_datagen = ImageDataGenerator(rescale=1./255)

# xy_train = train_datagen.flow_from_directory(
#     './_data/cat_dog/training_set',
#     target_size=(150,150),
#     batch_size=4000,
#     class_mode='categorical',
#     subset='training'
# )
# xy_test = train_datagen.flow_from_directory(
#     './_data/cat_dog/training_set',
#     target_size=(150, 150),
#     batch_size=1000,
#     class_mode='categorical',
#     subset='validation'
# )

# np.save('./_save/_npy/k59_8_train_x.npy', arr=xy_train[0][0])
# np.save('./_save/_npy/k59_8_train_y.npy', arr=xy_train[0][1])
# np.save('./_save/_npy/k59_8_test_x.npy', arr=xy_test[0][0])
# np.save('./_save/_npy/k59_8_test_y.npy', arr=xy_test[0][1])

# Found 6404 images belonging to 1 classes.
# Found 1601 images belonging to 1 classes.

x_train = np.load('./_save/_npy/k59_8_train_x.npy')
y_train = np.load('./_save/_npy/k59_8_train_y.npy')
x_test = np.load('./_save/_npy/k59_8_test_x.npy')
y_test = np.load('./_save/_npy/k59_8_test_y.npy')
print(x_train.shape) #(4000, 150, 150, 3)
print(y_train.shape) #(4000, 1)
print(x_test.shape)  #(1000, 150, 150, 3)
print(y_test.shape)  #(1000, 1)

vgg16 = VGG19(weights='imagenet', include_top=False,input_shape=(32,32,3))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout,LSTM, BatchNormalization, GlobalAveragePooling2D, Activation
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
model.add(Dense(1, activation= 'sigmoid'))

model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['acc'])
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=5, mode='min', verbose=1)

# model.fit(x_train, y_train)
hist = model.fit(x_train, y_train, epochs=50, steps_per_epoch=32, # 160 / 5 = 32
                    validation_data=(x_train,y_train), callbacks=[es], validation_split=0.1
                    # validation_steps=4
)

loss = model.evaluate(x_test, y_test)
print('loss : ', loss[0])
print('acc : ', loss[1])


# loss : 0.7471811495872641
# acc : 0.8819291953356184

