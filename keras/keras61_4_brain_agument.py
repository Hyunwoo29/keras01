# 훈련대이터를 기존데이터 20% 더할 것!
# 성과비교
# save_dir도 temp에 넣을것
# 증폭데이터는 temp에 저장후 훈련끝난후 결과 본뒤 삭제

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.layers.convolutional import Conv2D

train_datagen = ImageDataGenerator(
    rescale=1./255, # 0~ 255 사이니까 0~1로 만들어주기위해 255로 나눠줌
    horizontal_flip=True, # 좌우반전 True
    vertical_flip=True,
    width_shift_range=0.1, # 상하좌우 이동가능비율
    height_shift_range=0.1, # 상하좌우 이동가능비율
    rotation_range=5, # 회전 제한 각도(0~ 180)
    zoom_range=1.2, # 확대축소비율 (1.2 =  120%)
    shear_range=7,
    fill_mode='nearest' # nearest: 근접   근접일시 어느정도 매칭을 시켜라
)
# test_datagen = ImageDataGenerator(rescale=1./255)

xy_train = train_datagen.flow_from_directory(
    './_data/brain/train',
    target_size=(150, 150),
    batch_size=160,
    class_mode='binary'
)
xy_test = train_datagen.flow_from_directory(
    './_data/brain/test',
    target_size=(150, 150),
    batch_size=100,
    class_mode='binary'
)

x_train = xy_train[0][0]
y_train = xy_train[0][1]
x_test = xy_test[0][0]
y_test = xy_test[0][1]

augment_size = 192

randidx = np.random.randint(x_train.shape[0], size=augment_size)
# ic(x_train.shape[0]) # 60000
# ic(randidx) # randidx: array([59037, 19030, 23944, ..., 55441, 23621, 55814])
# ic(randidx.shape) # ic| randidx.shape: (40000,)

x_augmented = x_train[randidx].copy()  # 메모리 공유될 확률이 있기때문에 .copy()를 넣어줌
y_augmented = y_train[randidx].copy()

x_augmented = x_augmented.reshape(x_augmented.shape[0], 150, 150, 3)
x_train = x_train.reshape(x_train.shape[0], 150, 150, 3)
x_test = x_test.reshape(x_test.shape[0], 150, 150, 3)


x_augmented = train_datagen.flow(x_augmented, np.zeros(augment_size),
                                 batch_size=augment_size, shuffle=False,
                                 save_to_dir='./temp/'
                                ).next()[0]

x_train = np.concatenate((x_augmented, x_train))
y_train = np.concatenate((y_augmented, y_train))

print(x_train.shape)
# 2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

model = Sequential()
model.add(Conv2D(32, (2,2), input_shape=(150, 150, 3)))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

# 3. 컴파일, 훈련
model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['acc'])

# model.fit(x_train, y_train)
hist = model.fit(x_train, y_train, epochs=100, steps_per_epoch=32, # 160 / 5 = 32
                    validation_data=(x_train,y_train), validation_split=0.1
                    # validation_steps=4
)
# acc = hist.history['acc']
# val_acc = hist.history['val_acc']
# loss = hist.history['loss']
# val_loss = hist.history['val_loss']
result = model.evaluate(x_test, y_test)
print('loss : ', result[0])
print('acc : ', result[1])


# loss :  17.894987106323242
# acc :  0.5

#증폭
# loss :  2.2820422649383545
# acc :  0.5299999713897705