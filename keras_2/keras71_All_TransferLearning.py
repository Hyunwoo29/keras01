# pre-trained model

from tensorflow.keras.applications import VGG16, VGG19, Xception
from tensorflow.keras.applications import ResNet101, ResNet101V2, ResNet152,ResNet152V2
from tensorflow.keras.applications import ResNet50, ResNet50V2
from tensorflow.keras.applications import MobileNet, MobileNetV2, MobileNetV3Large,MobileNetV3Small
from tensorflow.keras.applications import DenseNet121, DenseNet169, DenseNet201
from tensorflow.keras.applications import NASNetLarge, NASNetMobile
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB2

model = ResNet152V2()

model.trainable = False

model.summary()

print('전체 가중치 개수 : ',len(model.weights))
print('훈련가능 가중치 개수 : ',len(model.trainable_weights))

# Xceptiopn()
# Total params: 22,910,480
# Trainable params: 0
# Non-trainable params: 22,910,480
# __________________________________________________________________________________________________
# 236
# 0

# ResNet101
# Total params: 44,707,176
# Trainable params: 0
# Non-trainable params: 44,707,176
# __________________________________________________________________________________________________
# 전체 가중치 개수 :  626
# 훈련가능 가중치 개수 :  0

# ResNet101V2
# Total params: 44,675,560
# Trainable params: 0
# Non-trainable params: 44,675,560
# __________________________________________________________________________________________________
# 전체 가중치 개수 :  544
# 훈련가능 가중치 개수 :  0


# ResNet152
# Total params: 60,419,944
# Trainable params: 0
# Non-trainable params: 60,419,944
# __________________________________________________________________________________________________
# 전체 가중치 개수 :  932
# 훈련가능 가중치 개수 :  0