# 실습
# 카테고리컬 시그모이드 쓸것
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

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten,  MaxPooling2D, Dropout

model = Sequential()
model.add(Conv2D(162, (2,2), input_shape=(150, 150, 3)))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation= 'relu'))
model.add(Dense(1, activation='sigmoid'))

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
print('val_acc : ', loss[-1])
