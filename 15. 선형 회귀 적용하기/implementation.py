# 15장 선형 회귀 적용하기

# 데이터 확인
# 주변 환경 변수를 이용한 보스턴 집값 예측
# 506 (0 ~ 505), 14 (0 ~ 13)
# 0, CRIM       : 인구 1인당 범죄 발생 수
# 1, ZN         : 25,000평방 피트 이상의 주거 구역 비중
# 2, INDUS      : 소매업 외 상업이 차지하는 면적 비율
# 3, CHAS       : 찰스강 위치 변수 (1: 강 주변, 0: 이외)
# 4, NOX        : 일산화질소 농도
# 5, RM         : 집의 평균 방 수
# 6, AGE        : 1940년 이전에 지어진 비율
# 7, DIS        : 5가지 보스턴 시 고용 시설까지의 거리
# 8, RAD        : 순환고속도로의 접근 용이성
# 9, TAX        : $10,000당 부동산 세율 총계
# 10, PTRATIO   : 지역별 학생과 교사 비율
# 11, B         : 지역별 흑인 비율
# 12, LSTAT     : 급여가 낮은 직업에 종사하는 인구 비율(%)
# 13, 가격 (단위: $1,000)

from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import tensorflow as tf

seed = 0
np.random.seed(seed)
tf.random.set_seed(3)

df = pd.read_csv("C:/Github Projects/DL-for-All/0. data/housing.csv", delim_whitespace=True, header=None)
# print(df.info())

dataset = df.values
X = dataset[:, 0:13]
Y = dataset[:, 13]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=seed)

model = Sequential()
model.add(Dense(30, input_dim=13, activation="relu"))
model.add(Dense(6, activation="relu"))
model.add(Dense(1))
# 선형 회귀라서 출력층에 활성화 함수를 지정할 필요 없음.

model.compile(loss="mean_squared_error", optimizer="adam")
model.fit(X_train, Y_train, epochs=200, batch_size=10)

Y_prediction = model.predict(X_test).flatten()
# sorted_pred = dict()
for i in range(10):
    label = Y_test[i]
    prediction = Y_prediction[i]
    # sorted_pred[label] = prediction
    print("실제가격: {:.3f}, 예상가격: {:.3f}".format(label, prediction))

# import matplotlib.pyplot as plt

# sorted_pred = dict.fromkeys(Y_test[1:10], Y_prediction[1:10])

# index = sorted(sorted_pred.items())
# plt.scatter(np.arange(0, 10), index.keys(), c="red")
# plt.scatter(np.arange(0, 10), index.values(), c="blue")
# plt.legend()
# plt.show()
