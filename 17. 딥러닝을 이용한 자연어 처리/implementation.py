# Chapter 17. 딥러닝을 이용한 자연어 처리
# 애플의 시리(siri), 구글의 어시스턴트(assistant), 아마존의 알렉사(alexa), 네이버의 클로바(clova)
# 필수적으로 사람의 언어를 이해하는 능력을 갖춰야 하는 AI 비서들
# 자연어 처리 (Natural Language Processing, NLP)
# 대용량 데이터를 학습할 수 있게 되면서(딥러닝) 자연어 처리 연구가 활발해짐.
# 텍스트를 정제하는 전처리 과정이 필수적
# string or character -> int (or factor)


# 1. 텍스트의 토큰화
# 텍스트는 단어, 문장, 형태소로 나눌 수 있음. 이렇게 작게 나누어진 하나의 단위를 토큰(token)이라 함.
# 그러니까 우리는 토큰을 정의해서 토큰으로 나누어야 함. 이를 토큰화(tokenization)라고 함.

# 1-1. 나누기만 합니다.
# keras가 제공하는 text_to_word_sequence는 공백문자를 기준으로 텍스트를 자름을 알 수 있었음.
# TODO from tensorflow.keras.preprocessing.text import text_to_word_sequence

# TODO text = "해보지 않으면 해낼 수 없다"
# TODO result = text_to_word_sequence(text)
# TODO print(result)

# 1-2. 빈도를 체크합니다.
# Bag-of-Words: 텍스트를 단어별로 잘랐을 때 단어의 빈도를 체크하는 방식(기법)

# TODO from tensorflow.keras.preprocessing.text import Tokenizer

# TODO docs = ["먼저 텍스트의 각 단어를 나누어 토큰화합니다.", "텍스트의 단어로 토큰화해야 딥러닝에서 인식됩니다.", "토큰화한 결과는 딥러닝에서 사용할 수 있습니다."]

# TODO token = Tokenizer()
# TODO token.fit_on_texts(docs)
# TODO print(token.word_counts)
# 특이하게도 클래스가 OrderDict다.

# TODO print(token.document_count)
# 문장 수까지 알 수 있다. (이건 아마도 리스트를 넣은 거라 그 갯수를 센 거지 않을까 싶다)

# TODO print(token.word_docs)
# 단어가 몇 개의 문장(데이터)에서 나오는지 알려준다. class는 당연히 dict
# 난 또 해당 단어가 나온 문장을 가리키는 줄. 그러고 보니 그러면 dict가 엉망이 되겠구나.

# TODO print(token.word_index)
# 나누어진 단어별로 인덱스를 알려준다. (굳이 보여주는 이유는 잘 모르겠다.)

# 요즘 많이 하는 생각, 코드가 중간에 있는 이론들은 쥬피터를 이용하는 게 더 낮지 않았을까 하는 생각.
# 이거 또, tokenizer에 대한 데이터필드로 정리해야 하겠구만. 귀찮네.

# 통합
# TODO from tensorflow.keras.preprocessing.text import text_to_word_sequence

# TODO text = "해보지 않으면 해낼 수 없다"

# TODO result = text_to_word_sequence(text)
# TODO print("원문:", text, "토큰화:", result, sep="\n")

# TODO from keras.preprocessing.text import Tokenizer

# TODO docs = ["먼저 텍스트의 각 단어를 나누어 토큰화 합니다.", "텍스트의 단어로 토큰화 해야 딥러닝에서 인식됩니다.", "토큰화 한 결과는 딥러닝에서 사용할 수 있습니다."]

# TODO token = Tokenizer()
# TODO token.fit_on_texts(docs)

# TODO print(
#     "단어 카운트:",
#     token.word_counts,
#     "문장 카운트:",
#     token.document_count,
#     "각 단어가 몇 개의 문장에 포함되어 있는가:",
#     token.word_docs,
#     "각 단어에 매겨진 인덱스 값:",
#     token.word_index,
#     sep="\n",
# )


# 2. 단어의 원-핫 인코딩
# word -> int(eger list): 각 인덱스 + 1을 위치값으로 취급

# TODO from tensorflow.keras.preprocessing.text import Tokenizer

# TODO text = "오랫동안 꿈꾸는 이는 그 꿈을 닮아간다"

# TODO token = Tokenizer()
# TODO token.fit_on_texts([text])
# TODO print(token.word_index)
# 단어별로 달린 인덱스를 확인할 수 있음

# TODO x = token.texts_to_sequences([text])
# word_index의 인덱스를 반환하는 작업이라고 보면 됨.
# 이후 to_categorical 함수를 통해 category에 대한 one-hot encoding을 진행함.

# TODO from keras.utils import to_categorical

# TODO word_size = len(token.word_index) + 1
# 첫번째 자리 0만들기 때문에 하나 더 필요함 / 그리고 뒤에 함수가 값 이상하다고 돌아가지도 않음.
# 안 정해줘도 알아서 길이 체크하고 해서 하는 것을 굳이 정해주겠다고 해서 이 사단을 낸 거구나.
# num_classes는 단어의 갯수를 얘기하는 게 아니라 출력의 한 리스트 당 length를 말하는 거였구요.
# TODO x = to_categorical(x, num_calsses=word_size)
# TODO print(x)


# 3. 단어 임베딩 (word embedding)
# 원-핫 인코딩은 텍스트가 길어지고 단어가 많아짐에 따라 각 한칸이 그리고 새 단어에 대한 한 리스트가 늘어나는 식으로 그냥 주구장창 늘어나기만 함.

# 그래서 주어진 배열을 정해진 길이로 압축시키는 방법(단어 임베딩)을 이용하고자 함.
# 결과적으로 단어간 의미 분석을 통해 비슷한 단어는 비슷한 것으로 묶자.

# TODO from keras.layers import Embedding

# TODO model = Sequential()
# TODO model.add(Embedding(16, 4))
# 왜 매번 넣는 크기를 정하죠? mini batch 때문에?
# 다음의 예제를 보고 알았는데, input_length라는 건 input으로 들어오는 문장이 저마다 다른 길이를 갖고 있어서 그걸 맞춰줬을 때의 단어 갯수를 의미하는 거였음. 근데 word embedding에 padding까지 하고 나면 다 같은 length의 list가 들어가는데 굳이 맞춰줘야 하나 싶긴 함.


# 4. 텍스트를 읽고 긍정, 부정 예측하기

# 코드는 전체로 보자.
# default
import numpy as np
import tensorflow as tf

# preprocessing
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# model constructure
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Embedding

docs = ["너무 재밌네요", "최고예요", "참 잘 만든 영화예요", "추천하고 싶은 영화입니다.", "한 번 더 보고 싶네요", "글쎄요", "별로예요", "생각보다 지루하네요", "연기가 어색해요", "재미없어요"]
classes = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)

x = token.texts_to_sequences(docs)
print(x)
# 문장별로 사용한 단어의 수가 다르다(다를 수 있다)
# 그러나 우리는 input의 크기를 맞춰야 한다.
# 이때 이용하는 것이 패딩, 즉 0을 덧데는 것이다.

padded_x = pad_sequences(x, 4)
print(padded_x)
# max 크기로 맞추면 되지 않을까.
# 여기까지는 각 문장 당 쓰인 단어를 숫자로 치환하는 과정이었음.

word_size = len(token.word_index) + 1

model = Sequential()
model.add(Embedding(word_size, 8, input_length=4))
model.add(Flatten())
model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

model.fit(padded_x, classes, epochs=20)
print("\nAccuracy: %.4f" % model.evaluate(padded_x, classes)[1])
