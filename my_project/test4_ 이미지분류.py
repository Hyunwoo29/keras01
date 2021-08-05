import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
from icecream import ic
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from PIL import Image
import matplotlib.pyplot as plt

# get_file을 사용하면, 해당 URL의 파일을 food_photos라는 폴더에 다운로드 합니다.
# untar를 true로 해서 압축을 풀어주고, 경로를 받아와, 이 경로를 사용하기 쉽도록, pathlib 객체로 만듭니다.
# import pathlib
# dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/food_photos.tgz"
# data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
# data_dir = pathlib.Path(data_dir)

import pathlib

from tensorflow.python.keras import activations
data_dir = pathlib.Path('./_data/kfood')

# 해당 경로 내 모든 하부 경로의 jpg 파일을 검색해 리스트화, 길이 측정
image_count = len(list(data_dir.glob('*/*.jpg')))
# ic(image_count) # ic| image_count: 77726

# 밥 하위 사진들을 리스트로 만들어 첫번째 사진 출력
# rice = list(data_dir.glob('밥/*'))
# rece_img = Image.open(str(rice[0]))
# rece_img.show()

# 데이터 전처리 및 입력 파이프라인 작성
batch_size = 32
img_height = 150
img_width = 150

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=2000,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=1000,
  image_size=(img_height, img_width),
  batch_size=batch_size)

# ic(train_ds, val_ds)
# Found 62385 images belonging to 13 classes.
# Found 15590 images belonging to 13 classes.

class_names = train_ds.class_names
# ic(class_names)
# ic| class_names: ['구이', '국', '김치', '나물', '떡', '만두', '면', '밥', '볶음', '음청류', '전', '죽', '튀김']

# 실제 이미지와 해당 이미지에 대한 정답 레이블을 가져오는것.
for image_batch, labels_batch in train_ds:
    print(image_batch.shape)
    print(labels_batch.shape)
    break
# 1배치 32개, 150 * 150의 3채널 컬러 이미지임을 알수있다.
# (32, 150, 150, 3)
# (32,)

# 메소드 체이닝 : 해당 메소드를 실행하면, 해당 기능이 적용된 자기 자신의 객체를 반환
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixels values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image)) 

# num_classes = 13

# model = Sequential([
#   layers.experimental.preprocessing.Rescaling(1./255, input_shape=(150, 150, 3)),
#   layers.Conv2D(16, 3, padding='same', activation='relu'),
#   layers.MaxPooling2D(),
#   layers.Conv2D(32, 3, padding='same', activation='relu'),
#   layers.MaxPooling2D(),
#   layers.Conv2D(64, 3, padding='same', activation='relu'),
#   layers.MaxPooling2D(),
#   layers.Flatten(),
#   layers.Dense(128, activation='relu'),
#   layers.Dense(num_classes)
# ])

# model.compile(optimizer='adam',
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])

# # model.summary()
# Layer (type)                 Output Shape              Param #
# =================================================================
# rescaling_1 (Rescaling)      (None, 150, 150, 3)       0
# _________________________________________________________________
# conv2d (Conv2D)              (None, 150, 150, 16)      448
# _________________________________________________________________
# max_pooling2d (MaxPooling2D) (None, 75, 75, 16)        0
# _________________________________________________________________
# conv2d_1 (Conv2D)            (None, 75, 75, 32)        4640
# _________________________________________________________________
# max_pooling2d_1 (MaxPooling2 (None, 37, 37, 32)        0
# _________________________________________________________________
# conv2d_2 (Conv2D)            (None, 37, 37, 64)        18496
# _________________________________________________________________
# max_pooling2d_2 (MaxPooling2 (None, 18, 18, 64)        0
# _________________________________________________________________
# flatten (Flatten)            (None, 20736)             0
# _________________________________________________________________
# dense (Dense)                (None, 128)               2654336
# _________________________________________________________________
# dense_1 (Dense)              (None, 5)                 645
# =================================================================
# Total params: 2,678,565
# Trainable params: 2,678,565
# Non-trainable params: 0
# _________________________________________________________________


# # 훈련
# epochs=10
# history = model.fit(
#   train_ds,
#   validation_data=val_ds,
#   epochs=epochs
# )

# # 학습 결과 시각화
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

# loss: 0.1212 - accuracy: 0.9588 - val_loss: 3.0828 - val_accuracy: 0.5610

# 데이터 증강
# 전처리 레이어 중, 랜덤하게 좌우를 변형시키는 레이어, 0.1만큼 이미지를 랜덤하게 기울이는 레이어,
# 그리고 0.1만큼 랜덤하게 이미지를 확대하는 레이어를 붙여줍니다.
data_augmentation = keras.Sequential(
  [
    layers.experimental.preprocessing.RandomFlip("horizontal", 
                                                 input_shape=(img_height, 
                                                              img_width,
                                                              3)),
    layers.experimental.preprocessing.RandomRotation(0.1),
    layers.experimental.preprocessing.RandomZoom(0.1),
  ]
)
# 시각화 구현
# plt.figure(figsize=(10, 10))
# for images, _ in train_ds.take(1):
#   for i in range(9):
#     augmented_images = data_augmentation(images)
#     ax = plt.subplot(3, 3, i + 1)
#     plt.imshow(augmented_images[0].numpy().astype("uint8"))
#     plt.axis("off")
# plt.show()
model = Sequential([
  data_augmentation,
  layers.experimental.preprocessing.Rescaling(1./255),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(13, 'softmax')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])
# model.summary()

# Layer (type)                 Output Shape              Param #
# =================================================================
# sequential (Sequential)      (None, 150, 150, 3)       0
# _________________________________________________________________
# rescaling_1 (Rescaling)      (None, 150, 150, 3)       0
# _________________________________________________________________
# conv2d (Conv2D)              (None, 150, 150, 16)      448
# _________________________________________________________________
# max_pooling2d (MaxPooling2D) (None, 75, 75, 16)        0
# _________________________________________________________________
# conv2d_1 (Conv2D)            (None, 75, 75, 32)        4640
# _________________________________________________________________
# max_pooling2d_1 (MaxPooling2 (None, 37, 37, 32)        0
# _________________________________________________________________
# conv2d_2 (Conv2D)            (None, 37, 37, 64)        18496
# _________________________________________________________________
# max_pooling2d_2 (MaxPooling2 (None, 18, 18, 64)        0
# _________________________________________________________________
# dropout (Dropout)            (None, 18, 18, 64)        0
# _________________________________________________________________
# flatten (Flatten)            (None, 20736)             0
# _________________________________________________________________
# dense (Dense)                (None, 128)               2654336
# _________________________________________________________________
# dense_1 (Dense)              (None, 13)                1677
# =================================================================
# Total params: 2,679,597
# Trainable params: 2,679,597
# Non-trainable params: 0
# _________________________________________________________________

epochs = 15
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']
ic("acc : ", acc)
ic("val_acc : ", val_acc)
ic("loss : ", loss)
ic("val_loss : ", val_loss)

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# ic| "acc : ": 'acc : '
#     acc: [0.3808646500110626,
#           0.4925755560398102,
#           0.541543185710907,
#           0.5759022235870361,
#           0.6005222201347351,
#           0.6141856908798218,
#           0.6236684918403625,
#           0.6349933743476868,
#           0.6428903341293335,
#           0.6502426862716675,
#           0.6615034937858582,
#           0.6632334589958191,
#           0.6706017851829529,
#           0.674446165561676,
#           0.6765605807304382]
# ic| "val_acc : ": 'val_acc : '
#     val_acc: [0.4521048367023468,
#               0.5000320076942444,
#               0.5505222082138062,
#               0.581982433795929,
#               0.5786505937576294,
#               0.5901839137077332,
#               0.5909527540206909,
#               0.6237585544586182,
#               0.602998673915863,
#               0.60799640417099,
#               0.6179919242858887,
#               0.6494521498680115,
#               0.6427884697914124,
#               0.6010764241218567,
#               0.6207470893859863]
# ic| "loss : ": 'loss : '
#     loss: [1.8344110250473022,
#            1.5059434175491333,
#            1.3570891618728638,
#            1.2579965591430664,
#            1.1938440799713135,
#            1.1501721143722534,
#            1.1185132265090942,
#            1.087660551071167,
#            1.0599703788757324,
#            1.0429946184158325,
#            1.0154834985733032,
#            1.0006498098373413,
#            0.9848732948303223,
#            0.9717355966567993,
#            0.960348904132843]
# ic| "val_loss : ": 'val_loss : '
#     val_loss: [1.6793010234832764,
#                1.5154176950454712,
#                1.3499200344085693,
#                1.260359525680542,
#                1.279799222946167,
#                1.2344433069229126,
#                1.2515822649002075,
#                1.1564608812332153,
#                1.204939365386963,
#                1.199452519416809,
#                1.18075430393219,
#                1.0674901008605957,
#                1.084640622138977,
#                1.2408556938171387,
#                1.1567493677139282]

# food_url = "./_data/predict/00001.jpg"
# food_path = tf.keras.utils.get_file('predict', origin=food_url)
 
food_path = "./_data/predict/00006.jpg"

img = keras.preprocessing.image.load_img(
    food_path, target_size=(img_height, img_width)
)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

ic(
    "정답은 {} 카테고리와 {:.2f} 퍼센트로 확실합니다."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)

# 면 이미지
# '정답은 면 카테고리와 71.83 퍼센트로 확실합니다.'
# '정답은 면 카테고리와 90.26 퍼센트로 확실합니다.'

# 콩나물국 이미지
# '정답은 국 카테고리와 51.31 퍼센트로 확실합니다.'

# 떡 이미지
# '정답은 떡 카테고리와 86.91 퍼센트로 확실합니다.'

# 김치 이미지
# '정답은 김치 카테고리 와 73.22 퍼센트로 확실합니다.'

# 전 이미지
# '정답은 전 카테고리와 76.76 퍼센트로 확실합니다.'

# 구이 이미지
# '정답은 구이 카테고리 와 99.28 퍼센트로 확실합니다.'
