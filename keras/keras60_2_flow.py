from tensorflow.keras.datasets import fashion_mnist
import numpy as np
from tensorflow.python.keras.backend import zeros
from icecream import ic
(x_train, y_train), (x_test,y_test) = fashion_mnist.load_data()

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255, # 0~ 255 사이니까 0~1로 만들어주기위해 255로 나눠줌
    horizontal_flip=True, # 수평이동
    vertical_flip=False,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=0.1,
    shear_range=0.5,
    fill_mode='nearest' # nearest: 근접   근접일시 어느정도 매칭을 시켜라
)


augment_size = 40000

randidx = np.random.randint(x_train.shape[0], size=augment_size)
# ic(x_train.shape[0]) # 60000
# ic(randidx) # randidx: array([59037, 19030, 23944, ..., 55441, 23621, 55814])
# ic(randidx.shape) # ic| randidx.shape: (40000,)

x_augmented = x_train[randidx].copy()  # 메모리 공유될 확률이 있기때문에 .copy()를 넣어줌
y_augmented = y_train[randidx].copy()

x_augmented = x_augmented.reshape(x_augmented.shape[0], 28, 28, 1)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)


x_augmented = train_datagen.flow(x_augmented, np.zeros(augment_size),
                                 batch_size=augment_size, shuffle=False).next()[0]

ic(x_augmented.shape) # ic| x_augmented.shape: (40000, 28, 28, 1)

x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))

ic(x_train.shape, y_train.shape) # ic| x_train.shape: (100000, 28, 28, 1), y_train.shape: (100000,)

