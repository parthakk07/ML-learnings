# %%
import numpy as np
import pandas as pd

# %%
df = pd.read_csv("/home/parthak/Videos/autoYT/placement2.csv")
df.sample(5)
# %%
import seaborn as sns

sns.scatterplot(x=df["cgpa"], y=df["package"])
# %%
x = df.drop(["package"], axis=1)
y = df["package"]
# %%
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(
    x, y, test_size=0.2, random_state=22
)
X_train
# %%
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
# %%
lr.fit(X_train, Y_train)
# %%
y_pred = lr.predict(X_test)
# %%
from sklearn.metrics import r2_score

# %%
r2_score(Y_test, y_pred)
# %%
# slope
m = lr.coef_
m
# %%
# y intrercepts
b = lr.intercept_
b
# %%
import matplotlib.pyplot as plt

plt.scatter(df["cgpa"], df["package"])
plt.plot(X_train, lr.predict(X_train), color="red")


# %%
# class SLR:
#     def __init__(self):
#         self.m = None
#         self.b = None

#     def fit(self, X_train, Y_train):
#         num = 0
#         den = 0

#         for i in range(X_train.shape[0]):
#             num = num + ((X_train[i] - X_train.mean()) * (Y_train[i] - Y_train.mean()))
#             den = den + ((X_train[i] - X_train.mean()) ** 2)

#         self.m = num / den
#         self.b = Y_train.mean() - (self.m * X_train.mean())
#         print(self.m)
#         print(self.b)

#     def predict(self, X_test):

#         return self.m * X_test + self.b


# %%
class SLR:
    def __init__(self):
        self.m = None
        self.b = None

    def fit(self, X_train, y_train):

        num = 0
        den = 0

        for i in range(X_train.shape[0]):
            num = num + ((X_train[i] - X_train.mean()) * (y_train[i] - y_train.mean()))
            den = den + ((X_train[i] - X_train.mean()) * (X_train[i] - X_train.mean()))

        self.m = num / den
        self.b = y_train.mean() - (self.m * X_train.mean())
        print(self.m)
        print(self.b)

    def predict(self, X_test):

        print(X_test)

        return self.m * X_test + self.b


# %%
lr = SLR()
lr.fit(X_train, Y_train)
# %%
