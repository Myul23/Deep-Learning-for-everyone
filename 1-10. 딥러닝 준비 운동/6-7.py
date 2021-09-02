# ch 6. 퍼셉트론
# 인공 신경망 (Artificial Neural Network)

# perceptron: 입력 값과 활성화 함수를 사용해 출력 값을 다음으로 넘기는 가장 작은 신경망 단위
# 퍼셉트론은 뉴런에서 착안한 구조로서 뉴런과 마찬가지로 하나로는 무엇도 이룰 수 없다. 이는 직선으로는 현실의 문제를 해결할 수 없는 것으로 나타낼 수 있다.
# ? Marvin Minsky, Perceptrons (퍼셉트론의 명확한 (선형적) 한계), 1969
# ! -> Multi-layer Perceptron


# ch 7. 다층 퍼셉트론
# 문제를 2차원에서 3차원(은닉층)으로 해결하자.
# -> PCA 설명 초반에 나오는 복잡한 차원을 저차원으로 줄이는 방식과 유사
# 그러니까 초기에 생각했던 perceptrons는 몸체를 기준으로 '1 -> 1 -> ... -> 1' 이런 식으로 구성했는데 1을 2로 만들어보자고 한 것.
# 퍼셉트론은 그 시점의 선형 식을 구성하고 활성화 함수를 거친 값들을 전달한다.

import numpy as np


w11 = np.array([-2, -2])
w12 = np.array([2, 2])
w2 = np.array([1, 1])
b1 = 3
b2 = -1
b3 = -1


def MLP(x, w, b):
    y = np.sum(w * x) + b
    if y <= 0:
        return 0
    else:
        return 1

def OR(x1, x2):
    return MLP(np.array([x1, x2]), w11, b1)

def NAND(x1, x2):
    return MLP(np.array([x1, x2]), w12, b2)

def AND(x1, x2):
    return MLP(np.array([x1, x2]), w2, b3)


def XOR(x1, x2):
    return AND(NAND(x1, x2), OR(x1, x2))


if __name__ == "__main__":
    for x in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        y = XOR(x[0], x[1])
        print(f"입력 값: {x}\t출력 값: {y}")
