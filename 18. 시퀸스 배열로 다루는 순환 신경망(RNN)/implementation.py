# Chapter 18. RNN
# 문장을 이해한다 = 문장에 쓰인 단어와 단어 간의 관계를 이해한다는 말.
# 지금까지의 학습은 위치는 고려했을지언정 단어 간 관계를 이해하지 못했음

# 이를 보완하고자 나온 것이 순환 신경망(RNN, Recurrent Neural Network)
# RNN은 각 단어에 대한 연산에서 앞선 단어를 이용함.(설명 보충 필요)
# 공백 문자를 기준으로 하는 단어 절단에 대하여 오늘 -> 주가가 -> 몇이야 방식으로 학습을 진행

# 널리 이용되는 건 LSTM (Long Short Term Memory)
# 한 층을 단어 수만큼 반복하는 RNN 특성상 가중치 소실이 일어나기 쉬움
# 반복되는 층에서 다음 반복으로 값을 넘길지 넘기지 않을지를 결정하게 해서 가중치 소실 발생 가능성을 줄임.

# 출력 형태에 대해 자유로움
# 다수 입력, 단일 출력: 모든 반복(RNN)을 끝낸 마지막 값만 다음 층으로 전달
# 단일 입력, 다수 출력: 보통 이미지와 같은 다차원 데이터를 통해 여러가지를 유추 혹은 예측
# 다수 입력, 다수 출력: 예) 번역, 질문에 답하기

# 1. LSTM을 이용한 로이터 뉴스 카테고리 분류하기
# 다시, 문장의 의미를 파악하는 것은 기본적으로 내용의 주제를 파악해야 함.
# 대용량 데이터를 이용한 RNN (시험) 학습

# TODO from keras.datasets import reuters
# TODO import numpy as np

# TODO (X_train, Y_train), (X_test, Y_test) = reuters.load_data(num_words=1000, test_split=0.2)
# 단어를 빈도순으로 정렬했을 때, 1000th까지

# TODO category = np.max(Y_train) + 1
# 출력에 대해선 범주화는 되어있는데, 원-핫 인코딩은 진행되지 않은 것
# TODO print(category, "카테고리")
# TODO print(len(X_train), "학습용 뉴스 기사")
# TODO print(len(X_test), "테스트용 뉴스 기사")
# 이미 데이터가 tokenizer.texts_to_sequences를 통해 변환되어 있음.
# TODO print(X_train[0])

# TODO from keras.preprocessing import sequence

# TODO x_train = sequence.pad_sequences(X_train, maxlen=100)
# TODO x_test = sequence.pad_sequences(X_test, maxlen=100)
# 문장 당 단어의 갯수를 100개로 맞춥니다. reshape과 비슷하게 필요하면 padding을 실시하고, 필요하지 않으면 데이터를 버립니다.

# TODO from keras.utils import to_categorical

# TODO y_train = to_categorical(Y_train)
# TODO y_test = to_categorical(Y_test)

# 모델 구성
# TODO from keras.models import Sequential
# TODO from keras.layers import Embedding, LSTM, Dense

# TODO model = Sequential()
# TODO model.add(Embedding(input_dim=1000, output_dim=100))
# TODO model.add(LSTM(100, activation="tanh"))
# TODO model.add(Dense(46, activation="softmax"))
# 얘는 그냥 softmax를 이용했네.

# TODO model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
# TODO history = model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=100, epochs=20)

# "Validationn Accuracy: %.4f" %
# TODO print(model.evaluate(x_test, y_test))


# * 통합

from keras.datasets import reuters

from keras.preprocessing import sequence
from keras.utils import to_categorical

from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


seed = 0
np.random.seed(seed)
tf.random.set_seed(3)

(X_train, Y_train), (X_test, Y_test) = reuters.load_data(num_words=1000, test_split=0.2)

x_train = sequence.pad_sequences(X_train, maxlen=100)
x_test = sequence.pad_sequences(X_test, maxlen=100)
y_train = to_categorical(Y_train)
y_test = to_categorical(Y_test)


model = Sequential()
model.add(Embedding(input_dim=1000, output_dim=100))
model.add(LSTM(100, activation="tanh"))
model.add(Dense(46, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=100, epochs=20)

print("Validationn Accuracy: %.4f" % model.evaluate(x_test, y_test)[1])


y_vloss = history.history["val_loss"]
y_loss = history.history["loss"]

x_len = np.arange(len(y_loss))
plt.plot(x_len, y_vloss, marker=".", c="red", label="Valtestset_loss")
plt.plot(x_len, y_loss, marker=".", c="blue", label="Trianset_loss")

plt.legend(loc="upper right")
plt.grid()
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()
# 0.7146


# 2. LSTM과 CNN의 조합을 이용한 영화 리뷰 분류하기
# 인터넷 영화 데이터베이스(Internet Movie Database, IMDB): 영화 정보, 출연진 정보, 개봉 정보, 영화 후기, 평점까지 2만 5000여 개
# 클래스: 긍정, 부정 (영화 평가) -> 원-핫 인코딩 과정이 필요 없다.

# 데이터셋(imdb), 패딩, Embedding -> Dropout -> Conv -> MaxPooling -> LSTM -> Dense -> Activation(???)

from keras.datasets import imdb
from keras.preprocessing import sequence

from keras.models import Sequential
from keras.layers import Embedding, Dropout, Conv1D, MaxPool1D, LSTM, Dense, Activation

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


seed = 0
np.random.seed(seed)
tf.random.set_seed(3)

(X_train, Y_train), (X_test, Y_test) = imdb.load_data(num_words=5000)

x_train = sequence.pad_sequences(X_train, maxlen=100)
x_test = sequence.pad_sequences(X_test, maxlen=100)


# train 전체 수로 test를 조정시켜버린 건가. 그러지 않고서 어떻게 저럴 수가 있지.
# maxs = []
# for i in np.arange(len(X_train)):
#     maxs.append(np.max(X_train[i]))
# print(np.max(maxs))

# maxs = []
# for i in np.arange(len(X_test)):
#     maxs.append(np.max(X_test[i]))
# print(np.max(maxs))


model = Sequential()
model.add(Embedding(5000, 100))
model.add(Dropout(0.5))
model.add(Conv1D(64, kernel_size=5, padding="valid", activation="relu"))
model.add(MaxPool1D(pool_size=4))
model.add(LSTM(55))
model.add(Dense(1))
model.add(Activation("sigmoid"))

# Conv1D에 대한 설명이 필요함, MaxPool1D도 설명이 필요함
# Conv2D랑 p.361 같은 모양이 비교 사진처럼 들어가는 게 좋겠지.

model.summary()
# 모델 가중치에 대한 상세한 summary를 볼 수 있음. (너무 좋다)

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
history = model.fit(x_train, Y_train, validation_data=(x_test, Y_test), batch_size=100, epochs=5)


print("Valtest Accuracy: %.4f" % model.evaluate(x_test, Y_test)[1])

y_loss = history.history["loss"]
y_vloss = history.history["val_loss"]

x_len = np.arange(len(y_loss))
plt.plot(x_len, y_vloss, marker=".", c="red", label="Valset_loss")
plt.plot(x_len, y_loss, marker=".", c="blue", label="Trainset_loss")

plt.legend(loc="upper right")
plt.grid()
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()
# 0.8516
