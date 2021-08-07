# Artificial Intelligence > Machine Learning > Deep Learning
# Supervised Learning vs Unsupervised Learning

# 4. 폐암 수술 환자의 생존율 예측하기 & 5. 딥러닝의 개괄 잡기

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# tensorflow가 비행기라면, keras는 조종사라고?

import numpy as np
import tensorflow as tf

np.random.seed(3)
tf.random.set_seed(3)

# Data set: 제대로된 데이터를 준비하는 일은 중요하다.
# 왜 또 웹에서 읽기가 안 되는 걸까. 그 repo는 열려있을 텐데.
# https://github.com/gilbutITbook/080228/blob/master/deeplearning/dataset/ThoraricSurgery.csv
# Data_set = np.loadtxt("C:/Github Projects/DL-for-All/0. data/ThoraricSurgery.csv", delimiter=",")
# 브로츠와프 의과대학(2013, 폴란드), 암 수술 환자의 수술 전 진단 데이터와 수술 후 생존 결과의 기록

# 470 x 18
X = Data_set[:, 0:17]
Y = Data_set[:, 17]

# model construct?
model = Sequential()
model.add(Dense(30, input_dim=17, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

# 다음 층으로 값을 어떻게 넘길 것이고, 어떻게 오차 값을 계산할 건지, 어떻게 오차를 줄여나갈지
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(X, Y, epochs=100, batch_size=10)


# 딥러닝을 위한 기초 수학
# 1. 일차 함수, 기울기와 y 절편
# 2. 이차 함수와 최솟값
# 3. 미분, 순간 변화율과 기울기
# 4. 편미분
# 5. 지수와 지수 함수
# 6. 시그모이드 함수
# 7. 로그와 로그 함수 (Logarithm == log)
