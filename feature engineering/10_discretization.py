# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.tree import DecisionTreeClassifier

# %%
df = pd.read_csv(
    "/home/parthak/Videos/autoYT/titanic/train.csv", usecols=["Age", "Fare", "Survived"]
)
df.sample(5)
# %%
df.dropna(inplace=True)
# %%
df.isnull().sum()
# %%
X = df.drop(columns=["Survived"])
Y = df["Survived"]
X
# %%
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.3, random_state=33
)
X_train
# %%
clf = DecisionTreeClassifier()
clf.fit(X_train, Y_train)
y_pred = clf.predict(X_test)
# %%
accuracy_score(y_pred, Y_test)
# %%
np.mean(cross_val_score(DecisionTreeClassifier(), X, Y, cv=10, scoring="accuracy"))
# %%
# 62 percent accuracy normal par now binning

# %%
kbin_age = KBinsDiscretizer(n_bins=10, encode="ordinal", strategy="quantile")
kbin_fare = KBinsDiscretizer(n_bins=10, encode="ordinal", strategy="quantile")

# %%
trf = ColumnTransformer([("first", kbin_age, [0]), ("second", kbin_fare, [1])])
# %%
X_train_transformed = trf.fit_transform(X_train)
X_test_transforemed = trf.transform(X_test)
# %%
trf.named_transformers_
# %%
trf.named_transformers_["first"].n_bins_
# %%
trf.named_transformers_["first"].bin_edges_
# %%


output = pd.DataFrame(
    {
        "age": X_train["Age"],
        "age_trf": X_train_transformed[:, 0],
        "fare": X_train["Fare"],
        "fare_trf": X_train_transformed[:, 1],
    }
)
# %%
output["age_labels"] = pd.cut(
    x=X_train["Age"], bins=trf.named_transformers_["first"].bin_edges_[0].tolist()
)
output["fare_labels"] = pd.cut(
    x=X_train["Fare"], bins=trf.named_transformers_["second"].bin_edges_[0].tolist()
)
# %%
output.sample(5)
# %%
clf = DecisionTreeClassifier()
clf.fit(X_train_transformed, Y_train)
y_pred2 = clf.predict(X_test_transforemed)
accuracy_score(y_pred2, Y_test)
# %%
np.mean(
    cross_val_score(
        DecisionTreeClassifier(),
        X_train_transformed,
        Y_train,
        cv=10,
        scoring="accuracy",
    )
)


# %%
# 65 percent accuracy increased 3 %
# function to do all this
#
#
#
# %%
def discetize(bins, strategy):
    kbin_age = KBinsDiscretizer(n_bins=bins, encode="ordinal", strategy=strategy)
    kbin_fare = KBinsDiscretizer(n_bins=bins, encode="ordinal", strategy=strategy)

    trf = ColumnTransformer([("first", kbin_age, [0]), ("second", kbin_fare, [1])])
    X_trf = trf.fit_transform(X)

    print(
        np.mean(
            cross_val_score(DecisionTreeClassifier(), X, Y, cv=10, scoring="accuracy")
        )
    )

    plt.figure(figsize=(14, 4))
    plt.subplot(121)
    plt.hist(X["Age"])
    plt.title("Before")

    plt.subplot(122)
    plt.hist(X_trf[:, 0], color="red")
    plt.title("After")

    plt.show()

    plt.figure(figsize=(14, 4))
    plt.subplot(121)
    plt.hist(X["Fare"])
    plt.title("before")

    plt.subplot(122)
    plt.hist(X_trf[:, 1], color="red")
    plt.title("fare")

    plt.show()


# %%
discetize(10, "quantile")


# %%
# binarizaion ka code
df = pd.read_csv("/home/parthak/Videos/autoYT/titanic/train.csv")[
    ["Age", "Fare", "SibSp", "Parch", "Survived"]
]
df.dropna(inplace=True)
df.head()
# %%
df["family"] = df["SibSp"] + df["Parch"]
# %%
df.drop(columns=["SibSp", "Parch"], inplace=True)
df.head()
# %%
X = df.drop(columns=["Survived"])
y = df["Survived"]
# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_train.sample(3)
# %%
# Without binarization

clf = DecisionTreeClassifier()

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy_score(y_test, y_pred)
# %%
np.mean(cross_val_score(DecisionTreeClassifier(), X, y, cv=10, scoring="accuracy"))
# %%
# 64 without binarization

# %%
# Applying Binarization

from sklearn.preprocessing import Binarizer

# %%
trf = ColumnTransformer(
    [("bin", Binarizer(copy=False), ["family"])], remainder="passthrough"
)

# %%
X_train_trf = trf.fit_transform(X_train)
X_test_trf = trf.transform(X_test)

# %%
pd.DataFrame(X_train_trf, columns=["family", "Age", "Fare"])
# %%
clf = DecisionTreeClassifier()
clf.fit(X_train_trf, y_train)
y_pred2 = clf.predict(X_test_trf)

accuracy_score(y_test, y_pred2)
# %%
X_trf = trf.fit_transform(X)
np.mean(cross_val_score(DecisionTreeClassifier(), X_trf, y, cv=10, scoring="accuracy"))

# %%
# 63 with binarization
#
