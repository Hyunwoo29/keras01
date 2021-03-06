# 2번 카피해서 복붙
# 딥하게 구성
# 2개의 모델을 구성하는데 하나는 기본적 오토인코더
# 다른 하나는 딥하게 만든 구성
# 2개의 성능 비교

# 앞뒤가 똑같은 오토인코더~
# 인코더 - 디코더로 구성되어있다.
# 인코더: 인지 네트워크, 입력을 내부 표현으로 변환한다.
# 디코더: 생성 네트워크, 내부 표현을 출력으로 변환한다.
# 결과물은 뿌옇게 나온다. (특성이 있다.)
# 특성이 약한걸 지운다.
from enum import auto
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.python.keras.engine import input_layer
from tensorflow.python.ops.variables import validate_synchronization_aggregation_trainable

# 1. 데이터
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.reshape(60000, 784).astype('float')/255
# x_train2 = x_train.reshape(60000, 784).astype('float')/255

x_test = x_test.reshape(10000,784).astype('float')/255

# 2. 모델
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, Dropout

def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Dense(units=hidden_layer_size, input_shape=(784,),
                    activation='relu'))
    model.add(Dense(units=784, activation='sigmoid'))
    return model

def autoencoder2(hidden_layer_size) :
    model = Sequential()
    model.add(Dense(units=hidden_layer_size, input_shape=(784,),
                    activation= 'relu'))
    model.add(Dense(units=1024, activation='relu'))
    model.add(Dense(units=512, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(units=512, activation='relu'))
    model.add(Dense(units=1024, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(units=784, activation='sigmoid'))
    return model


model = autoencoder(hidden_layer_size=154) # pca 95%
model.compile(optimizer='adam', loss='mse')
model.fit(x_train, x_train, epochs=10)

model2 = autoencoder2(hidden_layer_size=154) # pca 95%
model2.compile(optimizer='adam', loss='mse')
model2.fit(x_train, x_train, epochs=10)

output = model.predict(x_test)
output_1 = model2.predict(x_test)



from matplotlib import pyplot as plt
import random

fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10), (ax11, ax12, ax13, ax14, ax15)) = \
    plt.subplots(3, 5, figsize = (20, 7))

# 이미지 5개를 무작위로 고른다.
random_images = random.sample(range(output.shape[0]), 5)
random_images_1 = random.sample(range(output_1.shape[0]), 5)


# 원본(입력) 이미지를 맨 위에 그린다
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(28, 28), cmap = 'gray')
    if i == 0:
        ax.set_ylabel('INPUT', size = 20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 오토인코더가 출력한 이미지를 아래에 그린다.
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(output[random_images[i]].reshape(28, 28), cmap = 'gray')
    if i == 0:
        ax.set_ylabel('OUTPUT', size = 20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

for i, ax in enumerate([ax11, ax12, ax13, ax14, ax15]):
    ax.imshow(output_1[random_images_1[i]].reshape(28, 28), cmap = 'gray')
    if i == 0:
        ax.set_ylabel('OUTPUT_1', size = 20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()