from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd

datasets = load_iris()

irisDF = pd.DataFrame(data = datasets.data, columns=datasets.feature_names)

# print(irisDF)

kmean = KMeans(n_clusters=3, max_iter=300, random_state=66)
# max_iter는 에포랑 비슷한 개념, n_clusters는 3개의 라벨값을 뽑겠다.
kmean.fit(irisDF)

results = kmean.labels_
print(results)
print(datasets.target) # 원래 y값

irisDF['cluster'] = kmean.labels_    # 클러스터링해서 생성한 y값
irisDF['target'] = datasets.target    # 원래 y값

iris_results = irisDF.groupby(['target', 'cluster'])['sepal length (cm)'].count()
print(iris_results)