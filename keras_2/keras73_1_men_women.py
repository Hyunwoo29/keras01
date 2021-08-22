# 가장 잘나온 전이학습모델로
# 이 데이터를 학습시켜서 결과치 도출
# keras59번과의 성능 비교
from tensorflow.keras.applications import VGG19, EfficientNetB0
from tensorflow.keras.layers import Dense, Flatten,BatchNormalization, Activation, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from keras.datasets import cifar10,cifar100
from keras.applications.vgg19 import preprocess_input
from keras.optimizers import Adam

import numpy as np
x_train = np.load('./_save/_npy/k59_3_train_x.npy')
y_train = np.load('./_save/_npy/k59_3_train_y.npy')
x_test = np.load('./_save/_npy/k59_3_test_x.npy')
y_test = np.load('./_save/_npy/k59_3_test_y.npy')
print(x_train.shape) # (2648, 300, 300, 3)
print(y_train.shape) # (2648,)
print(x_test.shape) # (662, 300, 300, 3)
print(y_test.shape) # (662,)

x_train = x_train[:-1,:,:,:]
x_pred = x_train[-1,:,:,:].reshape(1,300,300,3)
y_train = y_train[:-1]

vgg16 = VGG19(weights='imagenet', include_top=False,input_shape=(32,32,3))

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
model.add(Dense(1, activation= 'sigmoid'))

# model = Sequential()
# model.add(Conv2D(64, (2,2), input_shape=(300, 300, 3)))
# model.add(Flatten())
# model.add(Dense(32, activation= 'relu'))
# model.add(Dense(1, activation='sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['acc'])
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1)

# model.fit(x_train, y_train)
hist = model.fit(x_train, y_train, epochs=50, steps_per_epoch=32, # 160 / 5 = 32
                    validation_data=(x_train,y_train), callbacks=[es], validation_split=0.1
                    # validation_steps=4
)
acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

results = model.evaluate(x_test, y_test)

# 위에거로 시각화 할것

print('acc: ', acc[-1])
# print('val_acc: ', val_acc[:-1])

# from tensorflow.keras.preprocessing import image
# test_image = image.load_img('./data/predict/test/00002341.jpg', target_size=(150, 150))
# test_image = image.img_to_array(test_image)
# test_image = np.expand_dims(test_image, axis=0)
# result = model.predict(test_image)


y_predict = model.predict(x_pred)
pred = (1-y_predict) * 100
print('남자 확률 : ',pred, '%')

# acc : 0.45598187923431396
# 남자 확률 : 45.5981879234314 %