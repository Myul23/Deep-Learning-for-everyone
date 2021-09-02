# ch 9. 신경망에서 딥러닝으로
# ? 기울기 소실 (Vanishing Gradient)
# 활성화 함수로 이용된 Sigmoid 함수의 미분 값이 0.3으로 1보다 작으면서 기울기 값을 계산하고자 앞층으로 올수록 작아질 수밖에 없었다.
# ! -> Activation function 교체

# Sigmoid에서 출발해서
# 위, 아래로 더 늘린 tanh (기울기 소실 문제 여전히 존재했음)
# 0보다 작은 값은 이용하지 않는 ReLU (popular)
# 0의 값을 만드는 걸 완화시키는 Softplus etc.


# ? 느린 적합
# 기존의 Gradient Descent 방식은 너무 천천히 움직이며 Local optima에 빠지기 쉬웠다.

# ! 확률적 경사 하강법 (Stochastic Gradient Descent, SGD)
# 전체 데이터를 사용하는 것이 아니라, 임의로 추출한 일부를  이용해 가중치를 update합니다. (결과적으로 더 빨리, 자주 udpate하게 한다)
# sampling한 데이터를 이용하기에 크게 움직이되 불안정해보일 수 있다.
# keras.optimizers.SGD(lr=0.1)

# ! 모멘텀, momentum
# 이전 update의 부호를 고려하며 가중치를 udpate한다.
# 관성의 방향을 고려해 진동과 폭을 줄이는 효과를 가져온다.
# keras.optimizers.SGD(lr=0.1, momentum=0.9)


# 네스테로프 모멘텀, NAG (Nesterov Momentum)
# 모멘텀이 이동시킬 방향으로 미리 이동해서 gradient를 계산, 불필요한 이동을 줄이는 효과
# keras.optimizers.SGD(lr=0.1, momentum=0.9, nesterov=True)

# Adagrad
# 이전 udpate의 크기를 생각해 적당히 움직인다.
# 변수의 업데이터가 잦으면 학습룰을 적게 하여 이동 보폭을 조절하는 방법
# keras.optimizers.Adagrad(lr=0.01)
# 추가 변수로는 epsilon(=1e-6), rho, decay

# RMSProp
# Adagrad의 보폭 민감도를 보완한 방법
# keras.optimizers.RMSProp(lr=0.001, rho=0.9, epsilon=1e-8, decay=0.0)

# Adam (popular)
# modentum과 RMSProp를 합친 방법
# keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0)
