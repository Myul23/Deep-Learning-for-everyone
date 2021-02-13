# NLP mini-project
# * kaggle: https://www.kaggle.com/samdeeplearning/deepnlp?select=Sheet_1.csv
# 이런 데이터도 있었다. https://www.kaggle.com/marklvl/sentiment-labelled-sentences-data-set

# 데이터 이해
# 문장 전처리 (int factor로 변환, 길이 맞추기-padding)
# word embedding
# training


# TODO 데이터 이해

# Sheet1.csv contains 80 user responses, in the responsetext column, to a therapy chatbot. Bot said: 'Describe a time when you have acted as a resource for someone else'.  User responded. If a response is 'not flagged', the user can continue talking to the bot. If it is 'flagged', the user is referred to help.
# * Sheet1.csv에는 치료 챗봇에 대한 80개의 사용자 응답이 응답 텍스트 열에 포함되어 있습니다. Bot은 다음과 같이 말했다: '당신이 다른 사람을 위한 자원 역할을 했던 때를 묘사하라.' 사용자가 응답했습니다. 응답에 플래그가 지정되지 않은 경우 사용자는 계속해서 봇과 대화할 수 있습니다. 플래그가 '플래그'된 경우 사용자에게 도움말을 참조합니다.

# Sheet2.csv contains 125 resumes, in the resumetext column. Resumes were queried from Indeed.com with keyword 'data scientist', location 'Vermont'. If a resume is 'not flagged', the applicant can submit a modified resume version at a later date. If it is 'flagged', the applicant is invited to interview.
# * 시트 2.csv에는 재개 텍스트 열에 125개의 이력서가 들어 있습니다. 이력서는 Indeed.com에서 키워드 '데이터 사이언티스트', 위치 'Vermont'로 조회되었다. 이력서에 플래그가 지정되지 않은 경우, 신청자는 나중에 수정된 이력서 버전을 제출할 수 있습니다. 플래그가 '플래그'된 경우 지원자는 인터뷰에 초대됩니다.

# 이걸로 뭘 해야 하죠?
# 새 재개/응답은 플래그가 지정되거나 플래그가 지정되지 않은 것으로 분류합니다.
# 여기에는 이력서와 응답이라는 두 가지 데이터 세트가 있습니다. 데이터를 열차 세트와 테스트 세트로 분할하여 분류기의 정확도를 테스트합니다. 두 문제에 동일한 분류기를 사용하기 위한 보너스 포인트.


# TODO 문장 전처리

import pandas as pd

df = pd.read_csv("C:/Github Projects/DL-for-All/17. 딥러닝을 이용한 자연어 처리/mini project/0. data/Sheet_1.csv")
print(df.head())
