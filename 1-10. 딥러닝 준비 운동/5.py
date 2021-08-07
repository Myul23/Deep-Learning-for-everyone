# ch 5. 참 거짓 판단 장치: 로지스틱 회귀
# 1. 로지스틱 회귀의 정의
# 로지스틱 회귀는 output을 0과 1로 제한한(+ logit 변환을 거친) 함수를 뜻하는 거였는데.
# 2. 시그모이드 함수 (로지스틱 함수)
# 시그모이드 함수가 logit 변환을 통해 output을 0과 1로 쉽게 제한하는 방법을 거치는 함수
# 사진 참고 https://inhovation97.tistory.com/6
# 3. 오차 공식
# 4. 로그 함수 (binary cross entropy를 그렸음)


# 5. 코딩으로 확인하는 로지스틱 회귀
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = [[2, 0], [4, 0], [6, 0], [8, 1], [10, 1], [12, 1], [14, 1]]
x_data = [i[0] for i in data]
y_data = [i[1] for i in data]

plt.scatter(x_data, y_data)
plt.xlim(0, 15)
plt.ylim(-0.1, 1.1)

a = 0
b = 0
lr = 0.05


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# loss = MSE, optimizer = SGD
for i in range(2001):
    for x_data, y_data in data:
        a_diff = x_data * (sigmoid(a * x_data + b) - y_data)
        b_diff = sigmoid(a * x_data + b) - y_data
        a = a - lr * a_diff
        b = b - lr * b_diff

    if i % 100 == 0:
        print("epoch=%d, 기울기=%0.4f, 절편=%.04f" % (i, a, b))

plt.scatter(x_data, y_data)
plt.xlim(0, 15)
plt.ylim(-0.1, 1.1)

x_range = np.arange(0, 15, 0.1)
plt.plot(np.arange(0, 15, 0.1), np.array([sigmoid(a * x + b) for x in x_range]))
plt.show()


# 6. 로지스틱 회귀에서 퍼셉트론으로

