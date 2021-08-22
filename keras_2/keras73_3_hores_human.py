# 실습
# 말과 사람 데이터셋으로 완성하시오!
# 카테고리컬과 소프트맥스 써라
from tensorflow.keras.applications import VGG19, EfficientNetB0
from tensorflow.keras.layers import Dense, Flatten,BatchNormalization, Activation, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from keras.datasets import cifar10,cifar100
from keras.applications.vgg19 import preprocess_input
from keras.optimizers import Adam
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# train_datagen = ImageDataGenerator(
#     rescale=1./255, # 0~ 255 사이니까 0~1로 만들어주기위해 255로 나눠줌
#     horizontal_flip=True, # 수평이동
#     vertical_flip=True,
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     rotation_range=5,
#     zoom_range=1.2,
#     shear_range=7,
#     fill_mode='nearest', # nearest: 근접   근접일시 어느정도 매칭을 시켜라
#     validation_split=.2
# )
# test_datagen = ImageDataGenerator(rescale=1./255)


# xy_train = train_datagen.flow_from_directory(
#     './_data/hores_human',
#     target_size=(300, 300),
#     batch_size=3000,
#     class_mode='categorical',
#     subset='training'
# )
# xy_test = train_datagen.flow_from_directory(
#     './_data/hores_human',
#     target_size=(300, 300),
#     batch_size=1000,
#     class_mode='categorical',
#     subset='validation'
# )


# np.save('./_save/_npy/k59_7_train_x.npy', arr=xy_train[0][0])
# np.save('./_save/_npy/k59_7_train_y.npy', arr=xy_train[0][1])
# np.save('./_save/_npy/k59_7_test_x.npy', arr=xy_test[0][0])
# np.save('./_save/_npy/k59_7_test_y.npy', arr=xy_test[0][1])

# Found 2016 images belonging to 3 classes.
# Found 504 images belonging to 3 classes.

x_train = np.load('./_save/_npy/k59_7_train_x.npy')
y_train = np.load('./_save/_npy/k59_7_train_y.npy')
x_test = np.load('./_save/_npy/k59_7_test_x.npy')
y_test = np.load('./_save/_npy/k59_7_test_y.npy')
print(x_train.shape) #(2016, 300, 300, 3)
print(y_train.shape) #(2016, 3)
print(x_test.shape)  #(504, 300, 300, 3)
print(y_test.shape)  #(504, 3)

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
model.add(Dense(3, activation= 'softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['acc'])
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1)

# model.fit(x_train, y_train)
hist = model.fit(x_train, y_train, epochs=50, steps_per_epoch=32, # 160 / 5 = 32
                    validation_data=(x_train,y_train), callbacks=[es], validation_split=0.1
                    # validation_steps=4
)

loss = model.evaluate(x_test, y_test)
print('category : ', loss[0])
print('acc : ', loss[1])

# category :  0.64231602111321
# acc :  0.891928213086712