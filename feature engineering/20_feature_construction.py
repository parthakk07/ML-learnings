# %%
import numpy as np
import pandas as pd

# %%
df = pd.read_csv("/home/parthak/Videos/autoYT/titanic/train.csv")[
    ["Age", "Pclass", "SibSp", "Parch", "Survived"]
]
df.sample(5)
# %%
df.shape
# %%
df.dropna(inplace=True)
# %%
x = df.drop(["Survived"], axis=1)
y = df["Survived"]
# %%
from sklearn.model_selection import train_test_split

X_train, X_trst, Y_train, Y_test = train_test_split(
    x, y, test_size=0.2, random_state=44
)
X_train
# %%
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

np.mean(cross_val_score(LogisticRegression(), x, y, cv=20, scoring="accuracy"))
# %%
# 69 percent accurate without feature cinstruction
#
x["family_size"] = x["SibSp"] + x["Parch"] + 1
x


# %%
def fun(num):
    if num == 1:
        return 0
    elif num > 1 and num <= 4:
        return 1
    else:
        return 2


# %%
x["family type"] = x["family_size"].apply(fun)
# %%
x.drop(["SibSp", "Parch", "family_size"], axis=1, inplace=True)
# %%
np.mean(cross_val_score(LogisticRegression(), x, y, scoring="accuracy", cv=20))
# %%
# 70 % this thime it increased the accuracy
#
# # feature splitting
df = pd.read_csv("/home/parthak/Videos/autoYT/titanic/train.csv")
df.sample(5)
# %%
df["Name"]
# %%

df["Title"] = df["Name"].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]


# %%
df
# %%
df[["Title", "Name"]]
# %%
(df.groupby("Title").mean()["Survived"]).sort_values(ascending=False)
# %%
df["Is_Married"] = 0
df["Is_Married"].loc[df["Title"] == "Mrs"] = 1
# %%
df
