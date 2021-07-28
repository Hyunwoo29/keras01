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
    fill_mode='nearest' # nearest: 근접   근접일시 어느정도 매칭을 시켜라
)
test_datagen = ImageDataGenerator(rescale=1./255)

xy_train = train_datagen.flow_from_directory(
    './_data/brain/train',
    target_size=(150, 150),
    batch_size=5,
    class_mode='binary'
)
# # print(xy_train[0][0]) # x값
# # print(xy_train[0][1]) # y값
# # print(xy_train[0][2]) # 없음
# print(xy_train[0][0].shape, xy_train[0][1].shape) #(5, 150, 150, 3) #(배치, 타겟사이즈, 타겟사이즈, 컬러라서3) (5,)

# print(xy_train[31][1]) #마지막 배치 y
# print(xy_train[32][1]) # 없음

# print(type(xy_train))
# print(type(xy_train[0]))
# print(type(xy_train[0][0]))
# print(type(xy_train[0][1]))
