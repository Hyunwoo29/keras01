from keras.models import Sequential
from keras.layers import MaxPooling2D
from keras.layers import Conv2D, Conv1D
from keras.layers import Activation, Dropout, Flatten, Dense
import numpy as np
import os
from icecream import ic
from sklearn.model_selection import train_test_split
from PIL import Image


categories = ["구이", "국", "김치", "나물", "떡", "만두", "면", "밥",
                "볶음", "음청류", "전", "죽", "튀김"]

nb_classes = len(categories)

image_w = 150
image_h = 150

x = np.load("./_save/_npy/project_x.npy",allow_pickle=True)
y = np.load("./_save/_npy/project_y.npy",allow_pickle=True)
# ic(x.shape, y.shape) 
# ic| x.shape: (89788, 255, 255, 3), y.shape: (89788, 13)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2)

x_train = x_train.astype("float") / 255
x_test  = x_test.astype("float")  / 255
# ic(x_train.shape, x_test.shape)
# ic| x_train.shape: (62180, 150, 150), x_test.shape: (15546, 150, 150)
x_train = x_train.reshape(62180, 150, 150, 1)
x_test = x_test.reshape(15546, 150, 150, 1)


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
model.add(Dense(13))   # 출력
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Dense(nb_classes))
model.add(Activation('softmax'))

# 모델 구축하기
model.compile(loss='categorical_crossentropy',   # 최적화 함수 지정
    optimizer='rmsprop',
    metrics=['accuracy'])
# 모델 확인
# print(model.summary())

model.fit(x_train, y_train, batch_size=32, epochs=10)


score = model.evaluate(x_test, y_test)
print('loss=', score[0])        # loss
print('accuracy=', score[1])    # acc

# 적용해볼 이미지 
test_image = './_data/unnamed.jpg'
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