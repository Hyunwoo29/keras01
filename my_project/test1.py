import numpy as np
from sklearn.model_selection import train_test_split
import os, glob
from PIL import Image
from icecream import ic

# 분류 대상 카테고리 선택하기
path_dir = "./_data/kfood"
categories = ["구이", "국", "김치", "나물", "떡", "만두", "면", "밥",
                "볶음", "음청류", "전", "죽", "튀김"]
# 카테고리 길이
classes = len(categories) # 13


# 이미지 크기 지정 그레이스케일로 할거라 세로픽셀수 * 가로픽셀수
image_w = 150
image_h = 150
pixels = image_w * image_h * 1

# RGB로 할거면 뒤에 색채널(3) 곱해서 3차원 배열로 저장.
# pixels = image_w * image_h * 3


# 이미지 데이터 읽어 들이기
X = []
Y = []
for idx, food in enumerate(categories):
    label = [0 for i in range(classes)]
    label[idx] = 1
    
    image_dir = path_dir + "/" + food
    files = glob.glob(image_dir+"/*.jpg")
    for i, f in enumerate(files):
        img = Image.open(f)
        img = img.convert("L")
        img = img.resize((image_w, image_h))
        data = np.asarray(img) # 복사본없이 이미지 저장
        X.append(data)
        Y.append(label)
        if i % 1000 == 0:
            print(food, " : ", data)

X = np.array(X)
Y = np.array(Y)



print('>>> data 저장중 ...')
# np.save("./_save/_npy/project_x.npy", arr=X)
# np.save("./_save/_npy/project_y.npy", arr=Y)
print("ok", len(X), len(Y)) # ok 77726 77726
            