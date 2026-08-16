# %%
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# %%
df = pd.read_csv("/home/parthak/Videos/autoYT/titanic/train.csv")[
    ["Age", "Pclass", "Fare", "Survived"]
]
df.sample(5)
# %%
X = df.drop(["Survived"], axis=1)
Y = df["Survived"]
# %%
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=22
)
# %%
# 3 compare with simple imputer
from sklearn.impute import SimpleImputer

si = SimpleImputer()
# %%
X_train_trf = si.fit_transform(X_train)
X_test_trf = si.transform(X_test)
# %%
clf = LogisticRegression()
clf.fit(X_train_trf, Y_train)
y_pred = clf.predict(X_test_trf)
# %%
from sklearn.metrics import accuracy_score

accuracy_score(y_pred, Y_test)
# %%
knn = KNNImputer()
# %%
X_train_transformed = knn.fit_transform(X_train)
X_test_transformed = knn.transform(X_test)
# %%
clf = LogisticRegression()
# %%
clf.fit(X_train_transformed, Y_train)
y_pred = clf.predict(X_test_transformed)
# %%
from sklearn.metrics import accuracy_score

accuracy_score(y_pred, Y_test)
# %%
# knn=74
# simpleimputer =76
