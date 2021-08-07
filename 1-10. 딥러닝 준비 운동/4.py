# ch 4. 오차 수정하기: 경사 하강법
import numpy as np
import matplotlib.pyplot as plt

# data = [[2, 81], [4, 93], [6, 91], [8, 97]]
# x = [i[0] for i in data]
# y = [i[1] for i in data]

# plt.figure(figsize=(8, 5))
# plt.scatter(x, y)
# plt.show()

# x_data = np.array(x)
# y_data = np.array(y)

# 최소 제곱법을 이용한 Gradient Descent
# a = 0
# b = 0
# lr = 0.03

# epochs = 2001
# for i in range(epochs):
#     y_pred = a * x_data + b
#     error = y_data - y_pred

#     a_diff = -(2 / len(x_data)) * sum(x_data * (error))

#     b_diff = -(2 / len(x_data)) * sum(error)

#     # simulative? update, 영어는 모르겠고, 한글은 동시에 바꾸기
#     a = a - lr * a_diff
#     b = b - lr * b_diff

#     if i % 100 == 0:
#         print("epoch=%d, \t기울기=%.4f, \t절편=%.04f" % (i, a, b))

# y_pred = a * x_data + b
# plt.scatter(x, y)
# plt.plot([min(x_data), max(x_data)], [min(y_pred), max(y_pred)])
# plt.show()


# 다중 선형 회귀
# y = a1x1 + a2x2 + ... + b
data = [[2, 0, 81], [4, 4, 93], [6, 2, 91], [8, 3, 97]]
x1 = [i[0] for i in data]
x2 = [i[1] for i in data]
y = [i[2] for i in data]


from mpl_toolkits import mplot3d

ax = plt.axes(projection="3d")
ax.scatter(x1, x2, y)

ax.set_xlabel("study_hours")
ax.set_ylabel("private_class")
ax.set_zlabel("Score")
plt.show()


x1_data = np.array(x1)
x2_data = np.array(x2)
y_data = np.array(y)

a1 = 0
a2 = 0
b = 0

lr = 0.02
epochs = 2001

for i in range(epochs):
    y_pred = a1 * x1_data + a2 * x2_data + b
    error = y_data - y_pred

    a1_diff = -(2 / len(x1_data)) * sum(x1_data * error)
    a2_diff = -(2 / len(x2_data)) * sum(x2_data * error)
    b_diff = -(2 / len(x1_data)) * sum(y_data - y_pred)

    a1 = a1 - lr * a1_diff
    a2 = a2 - lr * a2_diff
    b = b - lr * b_diff

    if i % 100 == 0:
        print("epoch=%.f, 기울기1=%0.4f, 기울기2=%0.4f, 절편=%0.4f" % (i, a1, a2, b))
