# 16장 이미지 인식의 꽃, CNN 익히기

# 데이터 확인
# MNIST, 70000 (60000, 10000)

# from keras.datasets import mnist
# from keras.utils import np_utils
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.callbacks import ModelCheckpoint, EarlyStopping

# import matplotlib.pyplot as plt
# import numpy as np
# import os
# import tensorflow as tf

# seed = 0
# np.random.seed(seed)
# tf.random.set_seed(3)

# plt.imshow(X_train[0], cmap="Greys")
# plt.show()

# import sys

# for x in X_train[0]:
#     for i in x:
#         sys.stdout.write("%d\t" % i)
#     sys.stdout.write("\n")

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 784).astype("float32") / 255
X_test = X_test.reshape(X_test.shape[0], 784).astype("float32") / 255
Y_train = np_utils.to_categorical(Y_train, 10)
Y_test = np_utils.to_categorical(Y_test, 10)

model = Sequential()
model.add(Dense(512, input_dim=784, activation="relu"))
model.add(Dense(10, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# MODEL_DIR = "./model/"
# if not os.path.exists(MODEL_DIR):
#     os.mkdir(MODEL_DIR)

# modelpath = "./model/{epoch:02d}-{val_loss:.4f}.hdf5"
# checkpointer = ModelCheckpoint(filepath=modelpath, monitor="val_loss", verbose=1, save_best_only=True)
# early_stopping_callback = EarlyStopping(monitor="val_loss", patience=10)

# history = model.fit(
#     X_train,
#     Y_train,
#     validation_data=(X_test, Y_test),
#     epochs=30,
#     batch_size=200,
#     verbose=0,
#     callbacks=[early_stopping_callback, checkpointer],
# )

# print("\n Test Accuracy: %.4f" % (model.evaluate(X_test, Y_test)[1]))

# y_vloss = history.history["val_loss"]
# y_loss = history.history["loss"]

# x_len = np.arange(len(y_loss))
# plt.plot(x_len, y_vloss, marker=".", c="red", label="Testset_loss")
# plt.plot(x_len, y_loss, marker=".", c="blue", label="Trainset_loss")

# plt.legend(loc="upper right")
# plt.grid()
# plt.xlabel("epoch")
# plt.ylabel("loss")
# plt.show()

# TODO CNN

from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping

import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

seed = 0
np.random.seed(seed)
tf.random.set_seed(3)

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype("float32") / 255
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype("float32") / 255
Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(28, 28, 1), activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

MODEL_DIR = "./model/"
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

modelpath = "./model/{epoch:02d}-{val_loss:.4f}.hdf5"
checkpointer = ModelCheckpoint(filepath=modelpath, monitor="val_loss", verbose=1, save_best_only=True)
early_stopping_callback = EarlyStopping(monitor="val_loss", patience=10)

history = model.fit(
    X_train,
    Y_train,
    validation_data=(X_test, Y_test),
    epochs=30,
    batch_size=200,
    verbose=0,
    callbacks=[early_stopping_callback, checkpointer],
)

print("\n Test Accuracy: %.4f" % (model.evaluate(X_test, Y_test)[1]))

y_vloss = history.history["val_loss"]
y_loss = history.history["loss"]

x_len = np.arange(len(y_loss))
plt.plot(x_len, y_vloss, marker=".", c="red", label="Testset_loss")
plt.plot(x_len, y_loss, marker=".", c="blue", label="Trainset_loss")

plt.legend(loc="upper right")
plt.grid()
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()
