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
)
test_datagen = ImageDataGenerator(rescale=1./255)
x_pred = train_datagen.flow_from_directory(
    './_data/predict/test',
    target_size=(300, 300),
    batch_size=100,
    class_mode='binary'
)
np.save('./_save/_npy/k59_3_pred_x.npy', arr=x_pred[0][0])