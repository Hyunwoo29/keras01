from scipy.sparse.construct import rand
from xgboost import XGBRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import r2_score
from icecream import ic
from sklearn.feature_selection import SelectFromModel

x, y = load_boston(return_X_y=True)
# ic(x.shape, y.shape) ic| x.shape: (506, 13), y.shape: (506,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=66
)

#  모델
model = XGBRegressor(n_jobs=8)

# 훈련
model.fit(x_train, y_train)

# 평가, 예측
score = model.score(x_test, y_test)
ic("model.socre : ", score)

thresholds = np.sort(model.feature_importances_)
ic(thresholds)

for thresh in thresholds:
    selection = SelectFromModel(model, threshold=thresh, prefit=True)

    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)

    # ic(select_x_train.shape)

    selection_model = XGBRegressor(n_jobs=-1)
    selection_model.fit(select_x_train, y_train)

    y_predict = selection_model.predict(select_x_test)

    score = r2_score(y_test, y_predict)

    ic("Thresh=%.3f, n=%d, R2: %.2f%%" %(thresh, select_x_train.shape[1],
        score*100))