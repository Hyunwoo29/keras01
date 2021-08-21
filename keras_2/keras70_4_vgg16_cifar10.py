# from tensorflow.keras.layers import Dense, Flatten
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.applications import VGG16
# from tensorflow.python.keras.applications.vgg19 import VGG19
# from keras.datasets import cifar10
# import numpy as np
# vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(32,32,3))

# (x_train, y_train), (x_test,y_test)= cifar10.load_data()
# vgg16.trainable=True # vgg 훈련을 동결한다.

# from tensorflow.keras.preprocessing.image import ImageDataGenerator

# train_datagen = ImageDataGenerator(
#     rescale=1./255, # 0~ 255 사이니까 0~1로 만들어주기위해 255로 나눠줌
#     horizontal_flip=True, # 수평이동
#     vertical_flip=False,
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     rotation_range=5,
#     zoom_range=0.1,
#     shear_range=0.5,
#     fill_mode='nearest' # nearest: 근접   근접일시 어느정도 매칭을 시켜라
# )

# # ic(x_train.shape[0]) # 50000

# augment_size = 50000

# randidx = np.random.randint(x_train.shape[0], size=augment_size)
# # ic(x_train.shape[0]) # 60000
# # ic(randidx) # randidx: array([59037, 19030, 23944, ..., 55441, 23621, 55814])
# # ic(randidx.shape) # ic| randidx.shape: (40000,)

# x_augmented = x_train[randidx].copy()  # 메모리 공유될 확률이 있기때문에 .copy()를 넣어줌
# y_augmented = y_train[randidx].copy()

# x_augmented = x_augmented.reshape(x_augmented.shape[0], 32, 32, 3)
# x_train = x_train.reshape(x_train.shape[0], 32, 32, 3)
# x_test = x_test.reshape(x_test.shape[0], 32, 32, 3)


# x_augmented = train_datagen.flow(x_augmented, np.zeros(augment_size),
#                                  batch_size=augment_size, shuffle=False,
#                                 #  save_to_dir='./temp/'
#                                 ).next()[0]

# # ic(x_augmented.shape) 
# # ic(x_augmented[0][0].shape)
# # ic(x_augmented[0][1].shape)
# # ic(x_augmented[0][1][:10])
# # ic(x_augmented[0][1][10:15])

# x_train = np.concatenate((x_train, x_augmented))
# y_train = np.concatenate((y_train, y_augmented))

# print(x_train.shape, y_train.shape) # ic| x_train.shape: (100000, 28, 28, 1), y_train.shape: (100000,)

# # 실습 1. x_augmented 10개와 원래 x_train 10개를 비교하는 이미지를 출력할것
# #  subplot(2, 10, ?) 사용
# # 2시까지

# # import matplotlib.pyplot as plt
# # plt.figure(figsize=(10,2))
# # for i in range(20):
# #     plt.subplot(2, 10, i+1)
# #     plt.axis('off')
# #     plt.imshow(x_train[0][i], cmap='gray')
# #     plt.imshow(x_augmented[0][i], cmap='gray')

# # plt.show()


# # 모델 완성!
# # 비교대상  loss, val_loss, acc, val_acc
# # 기존 fashon_mnist와 결과비교
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Conv2D, Flatten,  MaxPooling2D, Dropout

# model = Sequential()
# model.add(vgg16)
# model.add(Flatten())
# model.add(Dense(100))
# model.add(Dense(10))

# model.trainable=False # 천체 모델 훈련을 동결한다.


# model.summary()

# model.compile(loss = 'sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
# from tensorflow.keras.callbacks import EarlyStopping
# es = EarlyStopping(monitor='val_loss', patience=5, mode='min', verbose=1)

# # model.fit(x_train, y_train)
# hist = model.fit(x_train, y_train, epochs=100, steps_per_epoch=32, # 160 / 5 = 32
#                     validation_data=(x_train,y_train), callbacks=[es], validation_split=0.1
#                     # validation_steps=4
# )

# loss = model.evaluate(x_test, y_test)
# print(loss[0])
# print(loss[1])


# print(len(model.weights))       26 --> 30
# print(len(model.trainable_weights))  0 ---> 4