15장 선형 회귀 적용하기입니다.

이 장에서는 집값 예측을 위해 수집된 주변 환경 변수에 대한 데이터를 사용합니다. / 
이전과 마찬가지로 데이터를 확인하고 모델을 설정한 다음에 학습을 진행하도록 하겠습니다.

보스턴 데이터는 다음과 같은 14개의 변수를 가지고 있습니다. / 
0에서 12까지의 속성을 이용해 13, 가격을 예측하고자 했습니다.

info 함수를 통해 해당 데이터의 대략적인 형태를 둘러보면 / 13개의 열 중, 인덱스가 3, 8번에 해당하는 열은 범주형 데이터임을 알 수 있습니다. / 
앞선 데이터를 다시 둘러보면

주변에 강의 유무나 / 교통망에 대한 열임을 알 수 있습니다.

다음으로 모형의 구성입니다. / 
크게 3개의 계층을 통해 집값을 예측하고자 합니다. / 
속성은 13개였으므로 고려해야 할 input의 shape은 13이고 / Activation function은 ReLU를 공통적으로 사용합니다. / 
그리고 이전과 다른 점은 이번 출력이 연속형 데이터라는 점입니다. / 
굳이 어떤 class나 label로 출력할 필요가 없기 때문에 / 활성화 함수가 필요하지 않습니다.

나아가 출력이 class가 아니기 때문에 / 예측과 실제의 차이에 더 민감한 Mean Squared Error를 이용하고, Adam을 통해 최적화하고자 합니다. / 
이를 전체적인 코드로 보면

계층 구성에 필요한 함수들과 tensorflow를 이용하는데 기본적으로 사용되는 패키지들을 불러오고 / 훈련 데이터와 테스트 데이터를 쉽게 구분하는데 train_test_split 함수를 이용하고자 해당 함수를 불러옵니다.

이후 작업을 반복해서 확인할 때 편하고자 seed를 고정시키고 / 이용할 데이터셋을 불러와 훈련 데이터와 test 데이터로 나눕니다.

Sequential 함수를 통해 설계해둔 것처럼 크게 3개의 계층을 쌓습니다. / 
중요한 것은 전에도 얘기했다시피 출력이 집값으로 class나 label이 아닌 연속형 데이터이므로 출력층의 활성화 함수는 필요치 않습니다. / 
이어 loss를 계산할 방법을 지정하고 최적화 방법으로 adam을 지정하면 모델 구성이 완료됩니다. / 
이후 fit 함수를 통해 10개씩 전체적으로 200번의 학습을 진행합니다. / 
다음의 반복문으로 test를 통해 예측된 값을 일부 확인하면

몇 개의 큰 차이를 제외하면 / 대체로 비슷한 실제 값 근처로 예측됨을 알 수 있습니다.

이상으로 15장 발표를 마칩니다.
