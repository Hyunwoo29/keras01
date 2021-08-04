import numpy as np
from sklearn.model_selection import train_test_split
import os, glob
from PIL import Image

# 분류 대상 카테고리 선택하기
accident_dir = "./_data/kfood"
categories = ["구이", "국", "김치", "나물", "떡", "만두", "면", "밥",
                "볶음", "음청류", "전", "죽", "튀김"]

nb_classes = len(categories)

# 이미지 크기 지정
image_w = 150
image_h = 150
pixels = image_w * image_h * 3

# 이미지 데이터 읽어 들이기
X = []
Y = []
for idx, cat in enumerate(categories):
    label = [0 for i in range(nb_classes)]
    label[idx] = 1
    
    image_dir = accident_dir + "/" + cat
    files = glob.glob(image_dir+"/*.jpg")
    for i, f in enumerate(files):
        img = Image.open(f)
        img = img.convert("L")
        img = img.resize((image_w, image_h))
        data = np.asarray(img)
        X.append(data)
        Y.append(label)
        if i % 1000 == 0:
            print(cat, " : ", data)

X = np.array(X)
Y = np.array(Y)

# 학습 데이터와 테스트 데이터 구분
# x_train, x_test, y_train, y_test = train_test_split(X, Y)
# xy = (x_train, x_test, y_train, y_test)
print('>>> data 저장중 ...')
np.save("./_save/_npy/project_x.npy", arr=X)
np.save("./_save/_npy/project_y.npy", arr=Y)
print("ok", len(X), len(Y)) # ok 77726 77726
            