# Chapter 19. GAN, Auto-encoder
# 생성적 적대 신경망 (GAN, Generative Adversarial Networks): 가상의 이미지를 만들어내는 알고리즘
# GAN 내부에서 (적대적인) 경합을 진행
# (Ian Goodfellow said) 보다 진짜 같은 가짜를 만들고자 하는 위조지폐범과 진짜 같은 가짜를 판별하고자 하는 경찰의 경합
# 이때 위조지폐범, 즉 가짜를 만들어 내는 파트를 생성자 (Generator)
# (나머지) 경찰, 즉 진위를 가려내는 파트를 판별자 (Discriminator)
# DCGAN (Deep Convolutional GAN): Convolutional + GAN
# 초창기 GAN은 굳이 이미지를 타겟으로 하지 않아서 그랬는지? 아니면 CNN 개념이 나오기 전이라서 그랬는지 Convolutional 계층을 이용하지 않았음. 그래서 DCGAN이 등장하면서 GAN 알고리즘을 확립한 느낌


# 1. 가짜 제조 공장, 생성자
# optimizer X: GAN's Generator에는 학습 결과에 대한 판별이 필요하지 않으므로 최적화하거나 컴파일하는 과정이 없대.
# padding: 입력과 출력의 크기를 맞추기 위해서 패딩은 이용하지만, 같은 이유로 풀링은 이용하지 않음.
# batch normalization: 층이 늘어나도 안정적인 학습을 하기 위해서 다른 층의 전처리로 표준화 과정을 거침.
# activation: 연산 과정에선 relu를 이용하고, 판별자로 주기 전에 크기를 맞추는 과정에선 tanh를 이용해서 [-1, 1]로 맞추기.

from keras.models import Sequential
from keras.layers import Dense, LeakyReLU, BatchNormalization, Reshape, UpSampling2D, Conv2D, Activation

generator = Sequential()
generator.add(Dense(128 * 7 * 7, input_dim=100, activation=LeakyReLU(0.2)))
# 100개가 들어와서 (128 * 7 * 7)의 갯수로 내보내기
# GAN에서 ReLU를 이용하면 학습이 불안정(결과적으로 봤을 때 loss가 튄다든지, 최적화가 안 되고 멈춘다든지)해지는 경우가 많아, 조금 변형한 LeakyReLU를 이용
# LeakyReLU는 ReLU에서 x < 0 => 0이 되어 뉴런들이 일찍 소실되는 단점을 보완하기 위해, 0보다 작으면 들어온 인수(여기서는 0.2)를 곱해 보낸다.
generator.add(BatchNormalization())
generator.add(Reshape((7, 7, 128)))
# tensorflow에서 인식하는 차원은 n, 1D, 2D, color(3D)다.
generator.add(UpSampling2D())
# sub-sampling의 일종으로, (색채) 차원을 제외한 기본 이미지를 2배로 만드는 과정
generator.add(Conv2D(64, kernel_size=5, padding="same"))
generator.add(BatchNormalization())
generator.add(Activation(LeakyReLU(0.2)))
generator.add(UpSampling2D())
generator.add(Conv2D(1, kernel_size=5, padding="same", activation="tanh"))
# 연산으로 (색채) 차원 줄이기
# 사실 UpSmapling + Conv2D => Conv2DTranspose()로 하나로 표현할 수 있다.
# padding="same"으로 입력과 출력의 이미지 크기를 동일하게끔 합니다.
generator.summary()

# 작은 이미지를 늘려서 Convolutional 레이어를 지나치게 하는 것이 DCGAN의 특징.


# 2. 진위를 가려내는 장치, 판별자
# 판별자는 CNN 구조를 그대로 이용합니다. (이미지를 보고 클래스만 맞추면 되니까)
# 이전에 이용했던 CNN을 그대로 이용하지만, 결과적으로 학습해야 하는 건 생성자라 판별자는 학습하지 않는다.

from keras.models import Sequential
from keras.layers import Conv2D, Activation, LeakyReLU, Dropout, Flatten, Dense

discriminator = Sequential()
discriminator.add(Conv2D(64, input_shape=(28, 28, 1), kernel_size=5, strides=2, padding="same"))
# stride는 kernel window를 여러 칸 움직이게 해서 새로운 특징을 뽑아주는 효과가 생긴대.
# local적인 부분을 (약간이지만) 배제하기 때문인 것으로 파악됨.
# 맞았음. Dropout이나 Pooling처럼 새로운 필터를 적용한 효과를 내는 거래.
discriminator.add(Activation(LeakyReLU(0.2)))
discriminator.add(Dropout(0.3))
discriminator.add(Conv2D(128, kernel_size=5, strides=2, padding="same"))
discriminator.add(Activation(LeakyReLU(0.2)))
discriminator.add(Dropout(0.3))
discriminator.add(Flatten())
discriminator.add(Dense(1, activation="sigmoid"))
# 0 ~ 1의 값이어야 하고, 굳이 확률로 바꿀 필요가 없으니 sigmoid. (굳이 한다면 softmax도 가능)

discriminator.compile(loss="binary_crossentropy", optimizer="adam")
discriminator.trainable = False


# 3. 적대적 신경망 연결하기
# 실제 image -> discriminator 가중치 설정
# (->) input -> generator(input) -> discriminator(generator(input))
# 바로 위에 단계를 반복하면서 discriminator의 정답률이 0.5가 되면 학습을 종료시킴.

from keras.models import Input, Model

ginput = Input(shape=(100,))
dis_output = discriminator(generator(ginput))
gan = Model(ginput, dis_output)
gan.compile(loss="binary_crossentropy", optimizer="adam")


# 학습을 진행해줄 함수의 선언
from keras.datasets import mnist
import numpy as np


def gan_train(epoch, batch_size, saving_interval):
    # batch_size: 한 번에 몇 개의 실제 이미지와 몇 개의 가상 이미지를 판별자에 넣을 건지.
    # 그래서 모델에 2 * batch_size가 들어간다는 얘긴 아니겠지. 각각 batch_size / 2개씩이겠지?
    # 답은 두번째였구요.

    (X_train, _), (_, _) = mnist.load_data()
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype("float32")
    X_train = (X_train - 127.5) / 127.5

    true = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    for i in range(epoch):
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        imgs = X_train[idx]
        d_loss_real = discriminator.train_on_batch(imgs, true)
        # 딱 한 번 학습을 실시해 모델을 업데이트 개념상의 Gradient Descent다.

        noise = np.random.normal(0, 1, (batch_size, 100))
        # 그러나 생성자 input은 noise이고, tanh의 결과값에 따라가야 해서 정수가 아니네.
        gen_imgs = generator.predict(noise)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)

        d_loss = np.add(d_loss_real, d_loss_fake) * 0.5
        # 진짠데 가짜에 대한 loss와 가짠데 가짜에 대한 loss를 평균내면 판별자의 오차(loss)
        g_loss = gan.train_on_batch(noise, true)
        # 좋아, 이제 학습을 진행하자.

        print(f"epoch: {i}", "d_loss: %.4f" % d_loss, "g_loss: %.4f" % g_loss)


# 근데 이러면 마치 discriminator가 학습을 하지 않는다는 게 아니라 generator랑 같은 속도로 학습하게 하기 위해서 loss 계산 이외에 부분을 switch off 시켰다는 게 더 말이 됨.


# 4. 이미지의 특징을 추출하는 오토인코더
# Auto-Encoder (AE): GAN이 세상에 존재하지 않는 이미지를 만들어내는 거라면, AE는 입력 데이터의 특징을 (효율적으로) 담아낸 이미지를 만들어 냅니다.
# 다시, GAN이 random input에서 가중치를 통해 입력 이미지와 비슷한 형태를 만드는 거라면, AE는 입력 데이터의 특징을 가진 이미지를 나타내는 것.
# 따라서 GAN은 좀 더 명확한 이미지를 만들고, AE는 얼굴이라는 걸 알아볼 수 있을 정도의 특징만 나타내서 해상도가 낮은 것처럼 보일 수 있음.
# 개인적으로 GAN이 있을 법한 입력 데이터를 만드는 거라면, AE는 그림(사물의 특징)을 그리는 거랄까.

# AE: 영상 의학 분야 등 아직 데이터 수가 충분하지 않은 분야에서 사용될 수 있음.
# GAN은 가상의 것이므로 실제 데이터에 의존하는 분야에는 적합하지 않음.

# 확실히 이번 장부터는 다른 머신과 결합된 형태를 보여주고 있음.
# Encoder + Decoder의 형태로 이루어지며

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D

autoencoder = Sequential()
autoencoder.add(Conv2D(16, input_shape=(28, 28, 1), kernel_size=3, padding="same", activation="relu"))
autoencoder.add(MaxPooling2D(pool_size=2, padding="same"))
# 예상이 맞았다. sub-sampling에서 padding은 큰 의미를 갖지 않을 수도 있다.
autoencoder.add(Conv2D(8, kernel_size=3, padding="same", activation="relu"))
autoencoder.add(MaxPooling2D(pool_size=2, padding="same"))
autoencoder.add(Conv2D(8, kernel_size=3, strides=2, padding="same", activation="relu"))

autoencoder.add(Conv2D(8, kernel_size=3, padding="same", activation="relu"))
autoencoder.add(UpSampling2D())
autoencoder.add(Conv2D(8, kernel_size=3, padding="same", activation="relu"))
autoencoder.add(UpSampling2D())
autoencoder.add(Conv2D(16, kernel_size=3, activation="relu"))
autoencoder.add(UpSampling2D())
autoencoder.add(Conv2D(1, kernel_size=3, padding="same", activation="sigmoid"))

autoencoder.summary()
