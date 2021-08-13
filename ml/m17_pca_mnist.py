import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA

(x_train, _), (x_test, _) = mnist.load_data()

print(x_train.shape, x_test.shape)
# (60000, 28, 28) (10000, 28, 28)

x = np.append(x_train, x_test, axis=0)
# print(x.shape)
# (70000, 28, 28)
x = x.reshape(70000, 28* 28)
pca = PCA(n_components=28*28) # 컬럼을 9개로 압축하겠다. 라는뜻 (컬럼을 삭제하겠다는 의미가 아니라, 10개의 컬럼이면 9개로 압축을 하겠다는뜻. 다시 10개로 돌아갈수있다. 대신 손실률 0.00001정도 있다.)
x = pca.fit_transform(x)


pca_EVR = pca.explained_variance_ratio_
# print(pca_EVR)

# print(sum(pca_EVR))


cumsum = np.cumsum(pca_EVR)
# print(cumsum)
print(np.argmax(cumsum >= 0.999)+1)

# import matplotlib.pyplot as plt
# plt.plot(cumsum)
# plt.grid()
# plt.show()