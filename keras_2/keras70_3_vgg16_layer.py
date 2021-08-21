from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import VGG16
from tensorflow.python.keras.applications.vgg19 import VGG19

vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(100,100,3))
# model = VGG16()
# model = VGG19()
# model.trainable=False  # 전이학습 모델을 훈련시키지 않겠다. 위 weights='imagenet'값을 그대로 쓰겠다 라는말.(웨이트의 갱신이 없다.)
# model.summary()
# Layer (type)                 Output Shape              Param #
# =================================================================
# input_1 (InputLayer)         [(None, 224, 224, 3)]     0
# _________________________________________________________________
# block1_conv1 (Conv2D)        (None, 224, 224, 64)      1792
# _______________________________________________________________
# Total params: 138,357,544
# Trainable params: 138,357,544
# Non-trainable params: 0

# FC <- 용어정리해놔라!
# "완전 연결 되었다"는 뜻은 한 층(layer)의 모든 뉴런이 그 다음 층(layer)의 
# 모든 뉴런과 연결된 상태를 말한다. 1차원 배열의 형태로 평탄화된 행렬을 
# 통해 이미지를 분류하는데 사용되는 계층이다.


vgg16.trainable=False # vgg 훈련을 동결한다.

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(10))
model.add(Dense(1))

model.trainable=False # 천체 모델 훈련을 동결한다.


model.summary()
# print(len(model.weights))       26 --> 30
# print(len(model.trainable_weights))  0 ---> 4


###################################################################################

import pandas as pd
pd.set_option('max_colwidth', -1)
layers = [(layer,layer.name, layer.trainable) for layer in model.layers]
results = pd.DataFrame(layers, columns= ['Layer Type', 'Layer Name', 'Layer Trainable'])


print(results)
