import pandas as pd

df = pd.read_csv(
    "C:/Github Projects/DL-for-All/0. data/pima-indians-diabetes.csv",
    names=["pregnant", "plasma", "pressure", "thickness", "insulin", "BMI", "pedigree", "age", "class"],
)

print(df.head(), df.info(), df.describe(), sep="\n")


# checking tensorflow version 2.0.0에는 plotting용 package를 다운 받지 않았다.
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 12))
sns.heatmap(df.corr(), linewidths=0.1, vmax=0.5, cmap=plt.cm.gist_heat, linecolor="white", annot=True)
plt.show()

grid = sns.FacetGrid(df, col="class")
grid.map(plt.hist, "plasma", bins=10)
plt.show()


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import tensorflow as tf

# print(tf.__version__)

np.random.seed(3)
tf.random.set_seed(3)

df = np.loadtxt("C:/Github Projects/DL-for-All/0. data/pima-indians-diabetes.csv", delimiter=",")

# print(df.shape)

X = df[:, 0:8]
Y = df[:, 8]

model = Sequential()
model.add(Dense(12, input_dim=8, activation="relu"))
model.add(Dense(8, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(X, Y, epochs=200, batch_size=10)
print("\n Accuracy: %.4f" % (model.evaluate(X, Y)[1]))
# 0.7253
