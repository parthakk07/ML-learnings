# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import FunctionTransformer
from sklearn.tree import DecisionTreeClassifier

# %%
df = pd.read_csv(
    "/home/parthak/Videos/autoYT/titanic/train.csv", usecols=["Age", "Fare", "Survived"]
)
df.sample(5)

# %%
df["Age"].fillna(df["Age"].mean(), inplace=True)
# %%
df.isnull().sum()
# %%
x = df.drop(columns=["Survived"])
y = df["Survived"]
# %%
x
# %%
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(
    x, y, test_size=0.3, random_state=44
)
X_train
# %%
plt.figure(figsize=(14, 4))
plt.subplot(121)
sns.displot(X_train["Age"], kde=True)
plt.title("Age PDF")

plt.subplot(122)
stats.probplot(X_train["Age"], dist="norm", plot=plt)
plt.title("Age QQ Plot")

plt.show()
# %%
plt.figure(figsize=(14, 4))
plt.subplot(121)
sns.displot(X_train["Fare"], kde=True)
plt.title("Age PDF")

plt.subplot(122)
stats.probplot(X_train["Fare"], dist="norm", plot=plt)
plt.title("Age QQ Plot")

plt.show()
# %%
clf = LogisticRegression()
clf2 = DecisionTreeClassifier()
# %%
clf.fit(X_train, Y_train)
clf2.fit(X_train, Y_train)

y_pred = clf.predict(X_test)
y_pred1 = clf2.predict(X_test)

print("Accuracy LR", accuracy_score(Y_test, y_pred))
print("Accuracy DT", accuracy_score(Y_test, y_pred1))
# %%
trf = FunctionTransformer(func=np.log1p)
# %%
X_train_transformed = trf.fit_transform(X_train)
X_test_transformed = trf.transform(X_test)
# %%
clf = LogisticRegression()
clf2 = DecisionTreeClassifier()

clf.fit(X_train_transformed, Y_train)
clf2.fit(X_train_transformed, Y_train)

y_pred = clf.predict(X_test_transformed)
y_pred1 = clf2.predict(X_test_transformed)

print("Accuracy LR", accuracy_score(Y_test, y_pred))
print("Accuracy DT", accuracy_score(Y_test, y_pred1))

# %%
X_transformed = trf.fit_transform(x)

clf = LogisticRegression()
clf2 = DecisionTreeClassifier()

print("LR", np.mean(cross_val_score(clf, X_transformed, y, scoring="accuracy", cv=10)))
print("DT", np.mean(cross_val_score(clf2, X_transformed, y, scoring="accuracy", cv=10)))
# %%
plt.figure(figsize=(14, 4))

plt.subplot(121)
stats.probplot(X_train["Fare"], dist="norm", plot=plt)
plt.title("Fare Before Log")

plt.subplot(122)
stats.probplot(X_train_transformed["Fare"], dist="norm", plot=plt)
plt.title("Fare After Log")

plt.show()
# %%
plt.figure(figsize=(14, 4))

plt.subplot(121)
stats.probplot(X_train["Age"], dist="norm", plot=plt)
plt.title("Age Before Log")

plt.subplot(122)
stats.probplot(X_train_transformed["Age"], dist="norm", plot=plt)
plt.title("Age After Log")

plt.show()

# %%
trf2 = ColumnTransformer(
    [("log", FunctionTransformer(np.log1p), ["Fare"])], remainder="passthrough"
)

X_train_transformed2 = trf2.fit_transform(X_train)
X_test_transformed2 = trf2.transform(X_test)
# %%
clf = LogisticRegression()
clf2 = DecisionTreeClassifier()

clf.fit(X_train_transformed2, Y_train)
clf2.fit(X_train_transformed2, Y_train)

y_pred = clf.predict(X_test_transformed2)
y_pred2 = clf2.predict(X_test_transformed2)

print("Accuracy LR", accuracy_score(Y_test, y_pred))
print("Accuracy DT", accuracy_score(Y_test, y_pred2))
# %%
X_transformed2 = trf2.fit_transform(x)

clf = LogisticRegression()
clf2 = DecisionTreeClassifier()

print("LR", np.mean(cross_val_score(clf, X_transformed2, y, scoring="accuracy", cv=15)))
print(
    "DT", np.mean(cross_val_score(clf2, X_transformed2, y, scoring="accuracy", cv=15))
)


# %%
def apply_transform(transform):
    X = df.iloc[:, 1:3]
    y = df.iloc[:, 0]

    trf = ColumnTransformer(
        [("log", FunctionTransformer(transform), ["Fare"])], remainder="passthrough"
    )

    X_trans = trf.fit_transform(X)

    clf = LogisticRegression()

    print(
        "Accuracy", np.mean(cross_val_score(clf, X_trans, y, scoring="accuracy", cv=10))
    )

    plt.figure(figsize=(14, 4))

    plt.subplot(121)
    stats.probplot(X["Fare"], dist="norm", plot=plt)
    plt.title("Fare Before Transform")

    plt.subplot(122)
    stats.probplot(X_trans[:, 0], dist="norm", plot=plt)
    plt.title("Fare After Transform")

    plt.show()


# %%
apply_transform(np.sin)
# %%
