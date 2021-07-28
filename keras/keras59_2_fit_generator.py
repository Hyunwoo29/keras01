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
test_datagen = ImageDataGenerator(rescale=1./255)

xy_train = train_datagen.flow_from_directory(
    './_data/brain/train',
    target_size=(150, 150),
    batch_size=5,
    class_mode='binary'
)

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
hist = model.fit_generator(xy_train, epochs=50, steps_per_epoch=32, # 160 / 5 = 32
                    validation_data=xy_train,
                    validation_steps=4
)
acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

# 위에거로 시각화 할것

print('acc: ', acc[-1])
print('val_acc: ', val_acc[:-1])

