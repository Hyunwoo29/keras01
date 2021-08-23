# 앞뒤가 똑같은 오토인코더~
# 인코더 - 디코더로 구성되어있다.
# 인코더: 인지 네트워크, 입력을 내부 표현으로 변환한다.
# 디코더: 생성 네트워크, 내부 표현을 출력으로 변환한다.
# 결과물은 뿌옇게 나온다. (특성이 있다.)
# 특성이 약한걸 지운다.
from enum import auto
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.python.ops.variables import validate_synchronization_aggregation_trainable

# 1. 데이터
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.reshape(60000, 784).astype('float')/255
x_test = x_test.reshape(10000,784).astype('float')/255

# 2. 모델
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

input_img = Input(shape=(784,))
encoded = Dense(64, activation='relu')(input_img)

decoded = Dense(784, activation='sigmoid')(encoded)

autoencoder = Model(input_img, decoded)

# autoencoder.summary()
'''
Model: "model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         [(None, 784)]             0
_________________________________________________________________
dense (Dense)                (None, 64)                50240
_________________________________________________________________
dense_1 (Dense)              (None, 784)               50960
=================================================================
Total params: 101,200
Trainable params: 101,200
Non-trainable params: 0
'''
# 3. 컴파일, 훈련
# autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

autoencoder.fit(x_train, x_train, epochs=30, batch_size=128, validation_split=0.2)
#  fit에 y_train --> x_train 을 넣는이유야가 입력과 출력이 똑같기 때문이다.
# 즉 x와 y가 똑같다.

# 4. 예측
decoded_imgs = autoencoder.predict(x_test)

import matplotlib.pyplot as plt
n= 10
for i in range(n) :
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, i+1+n)
    plt.imshow(decoded_imgs[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()


# 디코더에 활성화함수 relu로 하면 relu는 0~ 무한대 이기때문에
# 값이 커진다. 그래서 그림이 더 뿌해진다(구려진다)