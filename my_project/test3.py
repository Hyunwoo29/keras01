from keras.models import Sequential
from keras.layers import MaxPooling2D
from keras.layers import Conv2D, Conv1D
from keras.layers import Activation, Dropout, Flatten, Dense
import numpy as np
import os
from icecream import ic
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping


categories = ["구이", "국", "김치", "나물", "떡", "만두", "면", "밥",
                "볶음", "음청류", "전", "죽", "튀김"]

nb_classes = len(categories)

image_w = 150
image_h = 150

x = np.load("./_save/_npy/project_x.npy",allow_pickle=True)
y = np.load("./_save/_npy/project_y.npy",allow_pickle=True)
# ic(x.shape, y.shape)  ic| x.shape: (77726, 150, 150), y.shape: (77726, 13)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state=66)

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
# x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
# x_predict = x_predict.reshape(x_predict.shape[0], x_predict.shape[1], 1)
# ic(x_train.shape, x_test.shape, x_predict.shape)
# # ic| x_train.shape: (66, 5, 1)
# #     x_test.shape: (29, 5, 1)
# #     x_predict.shape: (6, 4, 1)
# x_train = x_train.astype("float") / 255
# x_test  = x_test.astype("float")  / 255
# # ic(x_train.shape, x_test.shape)
# # ic| x_train.shape: (62180, 150, 150), x_test.shape: (15546, 150, 150)
# x_train = x_train.reshape(62180, 150, 150, 1)
# x_test = x_test.reshape(15546, 150, 150, 1)


# # 모델 구조 정의 
# model = Sequential()
# model.add(Conv2D(64, (3, 3), input_shape=(150, 150, 1), padding='same'))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

# model.add(Conv2D(32, (3, 3), padding='same'))
# model.add(Activation('relu'))

# model.add(Conv2D(32, (3, 3)))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

# # 전결합층
# model.add(Flatten())    # 벡터형태로 reshape
# model.add(Dense(13))   # 출력
# model.add(Activation('relu'))
# model.add(Dropout(0.25))

# model.add(Dense(nb_classes))
# model.add(Activation('softmax'))

# # 모델 구축하기
# model.compile(loss='categorical_crossentropy',   # 최적화 함수 지정
#     optimizer='rmsprop',
#     metrics=['accuracy'])
# # 모델 확인
# # print(model.summary())

# model.fit(x_train, y_train, batch_size=32, epochs=10)


# score = model.evaluate(x_test, y_test)
# print('loss=', score[0])        # loss
# print('accuracy=', score[1])    # acc