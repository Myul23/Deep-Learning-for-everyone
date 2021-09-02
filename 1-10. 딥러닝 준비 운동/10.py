# 딥러닝 기본기 다지기
# ch 10. 모델 설계하기

from keras import metrics
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import matplotlib.pyplot as plt


np.random.seed(3)
tf.random.set_seed(3)

Data_set = np.loadtxt("https://raw.githubusercontent.com/gilbutITbook/080228/master/deeplearning/dataset/ThoraricSurgery.csv", delimiter=",")

X = Data_set[:, 0:17]
Y = Data_set[:, 17]


# model set
model = Sequential()
model.add(Dense(30, input_dim=17, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

model.compile(loss="mean_squared_error", optimizer="adam", metrics=["accuracy"])
history = model.fit(X, Y, batch_size=10, epochs=100)

y_loss = history.history["loss"]
x_range = np.arange(len(y_loss))

plt.plot(x_range, y_loss)
plt.title("loss graph")
plt.show()

print("Accuracy: %.4f" % model.evaluate(X, Y)[1])


# ? 평균 제곱 계열
# mean_squared_error, MSE = mean(square(yt - y))
# mean_absolute_error, MAE = mean(abs(yt - y))
# mean_absolute_percentage_error, MAPE = mean(abs((yt - y) / yt)))
# mean_squared_logarithmic_error, MSLE? = mean(square((log(y) + 1) - (log(yt) + 1)))
# 평균 제곱 로그 오차, 실제 값과 예측 값에 로그를 적용한 값의 차이를 제곱한 값의 평균, 따라서 입력, 출력 모두에 log를 취한 후 구한 분산

# ? 교차 엔트로피 계열
# categorical_crossentropy, 범주형 교차 엔트로피 (일반적인 분류)
# binary_crossentropy, 이항 교체 엔트로피 (two classes)
