import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from icecream import ic
datasets = load_iris()
# print(datasets.keys())
# dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename'])

# print(datasets.target_names)
# ['setosa' 'versicolor' 'virginica']

x = datasets.data
y = datasets.target
# print(x.shape, y.shape)
# (150, 4) (150,)

# df = pd.DataFrame(x, columns=datasets.feature_names)
df = pd.DataFrame(x, columns=datasets["feature_names"]) # 이렇게 써두됨
# print(df)
'''
     sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
0                  5.1               3.5                1.4               0.2
1                  4.9               3.0                1.4               0.2
2                  4.7               3.2                1.3               0.2
3                  4.6               3.1                1.5               0.2
4                  5.0               3.6                1.4               0.2
..                 ...               ...                ...               ...
145                6.7               3.0                5.2               2.3
146                6.3               2.5                5.0               1.9
147                6.5               3.0                5.2               2.0
148                6.2               3.4                5.4               2.3
149                5.9               3.0                5.1               1.8
'''

# y컬럼 추가
df['Target'] = y
print(df.head())
'''
   sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)  Target
0                5.1               3.5                1.4               0.2       0
1                4.9               3.0                1.4               0.2       0
2                4.7               3.2                1.3               0.2       0
3                4.6               3.1                1.5               0.2       0
4                5.0               3.6                1.4               0.2       0
'''

ic("=========================상관계수 히트 맵====================================")
# ic(df.corr()) # corr은 판다스에서 제공
'''
ic| df.corr():                    sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)    Target
               sepal length (cm)           1.000000         -0.117570           0.871754          0.817941  0.782561
               sepal width (cm)           -0.117570          1.000000          -0.428440         -0.366126 -0.426658
               petal length (cm)           0.871754         -0.428440           1.000000          0.962865  0.949035
               petal width (cm)            0.817941         -0.366126           0.962865          1.000000  0.956547
               Target                      0.782561         -0.426658           0.949035          0.956547  1.000000
''' # sepal width 타겟 값이 안좋기 때문에 제거하는게 좋다.\

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.2)
sns.heatmap(data=df.corr(), square=True, annot=True, cbar=True)
plt.show()