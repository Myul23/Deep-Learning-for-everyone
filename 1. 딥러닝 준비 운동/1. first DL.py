# Artifitional Intelligence > Machine Learning > Deep Learning
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
# X = Data_set[:, 0:17]
# Y = Data_set[:, 17]

# model construct?
# model = Sequential()
# model.add(Dense(30, input_dim=17, activation="relu"))
# model.add(Dense(1, activation="sigmoid"))

# 다음 층으로 값을 어떻게 넘길 것이고, 어떻게 오차 값을 계산할 건지, 어떻게 오차를 줄여나갈지
# model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
# model.fit(X, Y, epochs=100, batch_size=10)

# 딥러닝을 위한 기초 수학
# 1. 일차 함수, 기울기와 y 절편
# 2. 이차 함수와 최솟값
# 3. 미분, 순간 변화율과 기울기
# 4. 편미분
# 5. 지수와 지수 함수
# 6. 시그모이드 함수
# 7. 로그와 로그 함수 (Logarithm == log)

# ch 3. 가장 훌륭한 예측선 긋기: 선형 회귀
# 최소제곱법: (이제야 묵은 떼가 씻기는 기분이다) 주어진 x의 값이 하나일 때 적용이 가능하다.
# 경사하강법은 mini-batch 가능.

# x = [2, 4, 6, 8]
# y = [81, 93, 91, 97]


# def top(x, mx, y, my):
#     d = 0
#     for i in range(len(x)):
#         d += (x[i] - mx) * (y[i] - my)
#     return d


# mx = np.mean(x)
# my = np.mean(y)

# divisor = sum([(i - mx) ** 2 for i in x])
# dividend = top(x, mx, y, my)

# a = dividend / divisor
# b = my - (mx * a)

# 출력으로 확인
# print(f"data:\n{x}\n{y}\n기울기: {a}, y 절편: {b}")

# 잘못 그은 선 바로잡기 -> MSE를 통한 loss의 측정
data = [[2, 81], [4, 93], [6, 91], [8, 97]]
x = [i[0] for i in data]
y = [i[1] for i in data]
# fake_a_b = [3, 76]


# def predict(x):
#     return fake_a_b[0] * x + fake_a_b[1]


# def mse(y, y_hat):
#     return ((y_hat - y) ** 2).mean()


# def mse_val(y, predict_result):
#     return mse(np.array(y), np.array(predict_result))


# predict_result = []
# for i in range(len(x)):
#     predict_result.append(predict(x[i]))
#     print("공부시간 = %.0f, 실제 점수 = %.0f, 예측 점수 = %.0f" % (x[i], y[i], predict(x[i])))
# print("MSE 최종값: " + str(mse_val(predict_result, y)))
# 기울기는 너무 커져도 loss를 높이고, 너무 작아져도 loss를 높이고

import matplotlib.pyplot as plt

# plt.figure(figsize=(8, 5))
# plt.scatter(x, y)
# plt.show()

x_data = np.array(x)
y_data = np.array(y)

# 최소 제곱법을 이용한 Gradient Descent
a = 0
b = 0
lr = 0.03

epochs = 2001
for i in range(epochs):
    y_pred = a * x_data + b
    error = y_data - y_pred

    a_diff = -(2 / len(x_data)) * sum(x_data * (error))

    b_diff = -(2 / len(x_data)) * sum(error)

    # simulative? update, 영어는 모르겠고, 한글은 동시에 바꾸기
    a = a - lr * a_diff
    b = b - lr * b_diff

    if i % 100 == 0:
        print("epoch=%d, \t기울기=%.4f, \t절편=%.04f" % (i, a, b))

y_pred = a * x_data + b
plt.scatter(x, y)
plt.plot([min(x_data), max(x_data)], [min(y_pred), max(y_pred)])
plt.show()
