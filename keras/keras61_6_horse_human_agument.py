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
# test_datagen = ImageDataGenerator(rescale=1./255)


xy_train = train_datagen.flow_from_directory(
    './_data/horse-or-human',
    target_size=(150, 150),
    batch_size=3000,
    class_mode='binary',
    subset='training',
    shuffle=False
)
xy_test = train_datagen.flow_from_directory(
    './_data/horse-or-human',
    target_size=(150, 150),
    batch_size=1000,
    class_mode='binary',
    subset='validation',
    shuffle=False # 마지막 사진을 내 사진으로 선택해야하기 때문에 셔플을 펄스로 넣음
)

x_train = xy_train[0][0]
y_train = xy_train[0][1]
x_test = xy_test[0][0]
y_test = xy_test[0][1]
print(x_train.shape) # (822, 150, 150, 3)
# augment_size = 822+int(822*0.2)

# randidx = np.random.randint(x_train.shape[0], size=augment_size)
# # ic(x_train.shape[0]) # 60000
# # ic(randidx) # randidx: array([59037, 19030, 23944, ..., 55441, 23621, 55814])
# # ic(randidx.shape) # ic| randidx.shape: (40000,)

# x_augmented = x_train[randidx].copy()  # 메모리 공유될 확률이 있기때문에 .copy()를 넣어줌
# y_augmented = y_train[randidx].copy()

# x_augmented = x_augmented.reshape(x_augmented.shape[0], 150, 150, 3)
# x_train = x_train.reshape(x_train.shape[0], 150, 150, 3)
# x_test = x_test.reshape(x_test.shape[0], 150, 150, 3)


# x_augmented = train_datagen.flow(x_augmented, np.zeros(augment_size),
#                                  batch_size=augment_size, shuffle=False,
#                                  save_to_dir='./temp/'
#                                 ).next()[0]

# x_train = np.concatenate((x_augmented, x_train))
# y_train = np.concatenate((y_augmented, y_train))

# print(x_train.shape)
# # x_train = x_train[:-1,:,:,:]
# # # x_pred = x_train[-1,:,:,:].reshape(1,300,300,3)
# # y_train = y_train[:-1]
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Conv2D, Flatten

# model = Sequential()
# model.add(Conv2D(64, (2,2), input_shape=(150, 150, 3)))
# model.add(Flatten())
# model.add(Dense(32, activation= 'relu'))
# model.add(Dense(1, activation='sigmoid'))

# model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['acc'])
# from tensorflow.keras.callbacks import EarlyStopping
# es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1)

# # model.fit(x_train, y_train)
# hist = model.fit(x_train, y_train, epochs=50, steps_per_epoch=32, # 160 / 5 = 32
#                     validation_data=(x_train,y_train), callbacks=[es], validation_split=0.1
#                     # validation_steps=4
# )
# # acc = hist.history['acc']
# # val_acc = hist.history['val_acc']
# # loss = hist.history['loss']
# # val_loss = hist.history['val_loss']

# results = model.evaluate(x_test, y_test)
# print('loss : ', results[0])
# print('acc : ', results[1])
# 위에거로 시각화 할것

# print('acc: ', acc[-1])
# print('val_acc: ', val_acc[:-1])

# from tensorflow.keras.preprocessing import image
# test_image = image.load_img('./data/predict/test/00002341.jpg', target_size=(150, 150))
# test_image = image.img_to_array(test_image)
# test_image = np.expand_dims(test_image, axis=0)
# result = model.predict(test_image)


# y_predict = model.predict(x_pred)
# pred = (1-y_predict) * 100
# print('남자 확률 : ',pred, '%')

# acc:  0.5331653952598572
# val_acc:  [0.9811320900917053, 0.99622642993927, 0.99245285987854, 0.99245285987854] 
# 남자 확률 :  [[49.314682]] %

# 정우성 사진넣고 확인
# 남자 확률 :  [[99.99984]] %