# # 59_3 npy를 이용하여 모델 완성

# # np.save('./_save/_npy/k59_3_train_x.npy', arr=xy_train[0][0])
# # np.save('./_save/_npy/k59_3_train_y.npy', arr=xy_train[0][1])
# # np.save('./_save/_npy/k59_3_test_x.npy', arr=xy_train[0][0])
# # np.save('./_save/_npy/k59_3_test_y.npy', arr=xy_train[0][1])

# import numpy as np
# x_train = np.load('./_save/_npy/k59_3_train_x.npy')
# y_train = np.load('./_save/_npy/k59_3_train_y.npy')
# x_test = np.load('./_save/_npy/k59_3_test_x.npy')
# y_test = np.load('./_save/_npy/k59_3_test_y.npy')
# # print(x_train.shape) # (160, 150, 150, 3)
# # print(y_train.shape) # (160,)
# # print(x_test.shape) # (160, 150, 150, 3)
# # print(y_test.shape) # (160,)

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Conv2D, Flatten

# model = Sequential()
# model.add(Conv2D(32, (2,2), input_shape=(150, 150, 3)))
# model.add(Flatten())
# model.add(Dense(1, activation='sigmoid'))

# model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['acc'])

# # model.fit(x_train, y_train)
# hist = model.fit(x_train, y_train, epochs=50, steps_per_epoch=32, # 160 / 5 = 32
#                     validation_data=(x_train,y_train)
#                     # validation_steps=4
# )
# acc = hist.history['acc']
# val_acc = hist.history['val_acc']
# loss = hist.history['loss']
# val_loss = hist.history['val_loss']

# # 위에거로 시각화 할것

# print('acc: ', acc[-1])
# print('val_acc: ', val_acc[:-1])


import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255, # 0~ 255 사이니까 0~1로 만들어주기위해 255로 나눠줌
    horizontal_flip=True, # 수평이동
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=1.2,
    shear_range=7,
    fill_mode='nearest', # nearest: 근접   근접일시 어느정도 매칭을 시켜라
    validation_split=.2,
)
test_datagen = ImageDataGenerator(rescale=1./255)


xy_train = train_datagen.flow_from_directory(
    './_data/men_women',
    target_size=(300, 300),
    batch_size=3000,
    class_mode='binary',
    subset='training',
    shuffle=False
)
xy_test = train_datagen.flow_from_directory(
    './_data/men_women',
    target_size=(300, 300),
    batch_size=1000,
    class_mode='binary',
    subset='validation',
    shuffle=False # 마지막 사진을 내 사진으로 선택해야하기 때문에 셔플을 펄스로 넣음
)


np.save('./_save/_npy/k59_3_train_x.npy', arr=xy_train[0][0])
np.save('./_save/_npy/k59_3_train_y.npy', arr=xy_train[0][1])
np.save('./_save/_npy/k59_3_test_x.npy', arr=xy_test[0][0])
np.save('./_save/_npy/k59_3_test_y.npy', arr=xy_test[0][1])
