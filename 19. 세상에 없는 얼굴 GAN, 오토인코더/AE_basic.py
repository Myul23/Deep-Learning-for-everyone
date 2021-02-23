from keras.datasets import mnist

from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt


np.random.seed(3)
tf.random.set_seed(3)


(X_train, _), (X_test, _) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype("float32") / 255
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype("float32") / 255

autoencoder = Sequential()
autoencoder.add(Conv2D(16, input_shape=(28, 28, 1), kernel_size=3, padding="same", activation="relu"))
autoencoder.add(MaxPooling2D(pool_size=2))
autoencoder.add(Conv2D(8, kernel_size=3, padding="same", activation="relu"))
autoencoder.add(MaxPooling2D(pool_size=2))
autoencoder.add(Conv2D(8, kernel_size=3, strides=2, padding="same", activation="relu"))

autoencoder.add(Conv2D(8, kernel_size=3, padding="same", activation="relu"))
autoencoder.add(UpSampling2D())
autoencoder.add(Conv2D(8, kernel_size=3, padding="same", activation="relu"))
autoencoder.add(UpSampling2D())
autoencoder.add(Conv2D(16, kernel_size=3, activation="relu"))
autoencoder.add(UpSampling2D())
autoencoder.add(Conv2D(1, kernel_size=3, padding="same", activation="sigmoid"))

# autoencoder.summary()

autoencoder.compile(loss="binary_crossentropy", optimizer="adam")
autoencoder.fit(X_train, X_train, validation_data=(X_test, X_test), batch_size=128, epochs=50)


# X_test의 일부분에 대해서 input과 output을 확인하는 작업
random_test = np.random.randint(X_test.shape[0], size=5)
ae_imgs = autoencoder.predict(X_test)

plt.figure(figsize=(7, 2))
for i, image_idx in enumerate(random_test):
    ax = plt.subplot(2, 7, i + 1)
    plt.imshow(X_test[image_idx].reshape(28, 28))
    ax.axis("off")

    ax = plt.subplot(2, 7, 7 + i + 1)
    plt.imshow(ae_imgs[image_idx].reshape(28, 28))
    ax.axis("off")
plt.show()
