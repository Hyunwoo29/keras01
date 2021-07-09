from sklearn.datasets import load_boston
datasets = load_boston()
x = datasets.data
y = datasets.target
print(x.shape)
print(y.shape)
# (506, 13)  input은 13개
# (506,)   output은 1개
print(datasets.feature_names)
#['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO'
#  'B' 'LSTAT'] 'B'는 흑인의 비율
print(datasets.DESCR)

# 완료하시오!