# %%

import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# %%
df = pd.read_csv("placement.csv")
df
# %%
df = df.iloc[:, 1:]
# %%
df.sample(5)  # %%
df.info()
# %%
df.describe()

# %%
df.sample(5)
# %%
plt.scatter(df["cgpa"], df["iq"], c=df["placement"])
# %%
x = df.iloc[:, 0:2]
y = df.iloc[:, -1]

## %%
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.1)

# %%
X_train
# %%
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# %%
X_train = scaler.fit_transform(X_train)
X_train
# %%
X_test = scaler.fit_transform(x_test)
X_train

# %%
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()

# %%
clf.fit(X_train, Y_train)
# %%
y_pred = clf.predict(X_test)
Y_test

# %%
from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_pred)

# %%
from mlxtend.plotting import plot_decision_regions

# %%
plot_decision_regions(X_train, Y_train.values, clf=clf, legend=2)

# %%
pickle.dump(clf, open("model.pkl", "wb"))
