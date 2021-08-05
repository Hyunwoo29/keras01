from keras.models import Sequential
from keras.layers import MaxPooling2D
from keras.layers import Conv2D, Conv1D
from keras.layers import Activation, Dropout, Flatten, Dense
import numpy as np
from icecream import ic
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.pyplot as plt

categories = ["구이", "국", "김치", "나물", "떡", "만두", "면", "밥",
                "볶음", "음청류", "전", "죽", "튀김"]

classes = len(categories)

image_w = 150 
image_h = 150

x = np.load("./_save/_npy/project_x.npy",allow_pickle=True)
y = np.load("./_save/_npy/project_y.npy",allow_pickle=True)
ic(x.shape, y.shape) 
# ic| x.shape: (77726, 150, 150), y.shape: (77726, 13)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state=66)
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, random_state=66, test_size=0.4)  
x_train = x_train.astype("float") / 255
x_test  = x_test.astype("float")  / 255
# ic(x_train.shape, x_test.shape)
# ic| x_train.shape: (62180, 150, 150), x_test.shape: (9327, 150, 150)

x_train = x_train.reshape(62180, 150, 150, 1)
x_test = x_test.reshape(9327, 150, 150, 1)


# 모델 구조 정의 
model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape=(150, 150, 1), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(32, (3, 3), padding='same'))
model.add(Activation('relu'))

model.add(Conv2D(32, (3, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# 전결합층
model.add(Flatten())    # 벡터형태로 reshape
model.add(Dense(13))   
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(13, activation='softmax'))

# 모델 구축하기
model.compile(loss='categorical_crossentropy',   # 최적화 함수 지정
    optimizer='rmsprop',
    metrics=['accuracy'])
# 모델 확인
# print(model.summary())

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1)
epochs= 10
history = model.fit(x_train, y_train, batch_size=128, epochs=epochs, callbacks=[es], validation_data=(x_train,y_train), validation_split=0.1)

score = model.evaluate(x_test, y_test)
print('loss=', score[0])        # loss
print('accuracy=', score[1])    # acc

# 학습 결과 시각화

# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']

# loss=history.history['loss']
# val_loss=history.history['val_loss']

# epochs_range = range(epochs)

# plt.figure(figsize=(8, 8))
# plt.subplot(1, 2, 1)
# plt.plot(epochs_range, acc, label='Training Accuracy')
# plt.plot(epochs_range, val_acc, label='Validation Accuracy')
# plt.legend(loc='lower right')
# plt.title('Training and Validation Accuracy')

# plt.subplot(1, 2, 2)
# plt.plot(epochs_range, loss, label='Training Loss')
# plt.plot(epochs_range, val_loss, label='Validation Loss')
# plt.legend(loc='upper right')
# plt.title('Training and Validation Loss')
# plt.show()

# 적용해볼 이미지 
test_image = './_data/predict/00004.jpg'
# 이미지 resize
img = Image.open(test_image)
img = img.convert("L")
img = img.resize((150,150))
data = np.asarray(img)
X = np.array(data)
X = X.astype("float") / 255
X = X.reshape(-1, 150, 150,1)
# 예측
pred = model.predict(X)  
result = [np.argmax(value) for value in pred]   # 예측 값중 가장 높은 클래스 반환
print('New data category : ',categories[result[0]])

# 라면 이미지
# New data category :  면
# loss= 1.900170087814331
# accuracy= 0.3833140432834625

# 콩나물국 이미지
# New data category :  구이
# loss= 2.355889320373535
# accuracy= 0.18022944033145905


# 떡이미지
# New data category :  구이
# loss= 2.3560049533843994
# accuracy= 0.18022944033145905


# 김치 이미지
# New data category :  구이
# loss= 2.3559324741363525
# accuracy= 0.18022944033145905

# 전 이미지
# New data category :  볶음
# loss= 2.114443778991699
# accuracy= 0.3262571096420288

# 구이 이미지
# New data category :  구이
# loss= 2.355865478515625
# accuracy= 0.18022944033145905

# adam을 사용하였을때
# loss= 1.9116727113723755
# accuracy= 0.37868553400039673

# rmsprop을 사용하였을때
# loss= 2.3559038639068604
# accuracy= 0.18022944033145905

# Layer (type)                 Output Shape              Param #
# =================================================================
# conv2d (Conv2D)              (None, 150, 150, 64)      640
# _________________________________________________________________
# activation (Activation)      (None, 150, 150, 64)      0
# _________________________________________________________________
# max_pooling2d (MaxPooling2D) (None, 75, 75, 64)        0
# _________________________________________________________________
# dropout (Dropout)            (None, 75, 75, 64)        0
# _________________________________________________________________
# conv2d_1 (Conv2D)            (None, 75, 75, 32)        18464
# _________________________________________________________________
# activation_1 (Activation)    (None, 75, 75, 32)        0
# _________________________________________________________________
# conv2d_2 (Conv2D)            (None, 73, 73, 32)        9248
# _________________________________________________________________
# max_pooling2d_1 (MaxPooling2 (None, 36, 36, 32)        0
# _________________________________________________________________
# dropout_1 (Dropout)          (None, 36, 36, 32)        0
# _________________________________________________________________
# flatten (Flatten)            (None, 41472)             0
# _________________________________________________________________
# dense (Dense)                (None, 13)                539149
# _________________________________________________________________
# activation_2 (Activation)    (None, 13)                0
# _________________________________________________________________
# dropout_2 (Dropout)          (None, 13)                0
# _________________________________________________________________
# dense_1 (Dense)              (None, 13)                182
# =================================================================
# Total params: 567,683
# Trainable params: 567,683
# Non-trainable params: 0
# _________________________________________________________________

