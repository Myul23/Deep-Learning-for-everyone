# ch 3. 가장 훌륭한 예측선 긋기: 선형 회귀
# 최소제곱법: (이제야 묵은 떼가 씻기는 기분이다) 주어진 x의 값이 하나일 때 적용이 가능하다.
# 경사하강법은 mini-batch 가능.


import numpy as np


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
fake_a_b = [3, 76]


def predict(x):
    return fake_a_b[0] * x + fake_a_b[1]


def mse(y, y_hat):
    return ((y_hat - y) ** 2).mean()


def mse_val(y, predict_result):
    return mse(np.array(y), np.array(predict_result))


predict_result = []
for i in range(len(x)):
    predict_result.append(predict(x[i]))
    print("공부시간 = %.0f, 실제 점수 = %.0f, 예측 점수 = %.0f" % (x[i], y[i], predict(x[i])))
print("MSE 최종값: " + str(mse_val(predict_result, y)))
# 기울기는 너무 커져도 loss를 높이고, 너무 작아져도 loss를 높이고
