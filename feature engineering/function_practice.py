# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score

# %%
df = pd.read_csv(
    "/home/parthak/Videos/autoYT/Social_Network_Ads.csv",
    usecols=["Age", "EstimatedSalary", "Purchased"],
)
df.sample(5)
# %%
X = df.iloc[:, 0:2]
Y = df.iloc[:, 2]
X

# %%
X.isnull().sum()
# %%
# age me nornal distribution hai !!
sns.kdeplot(X["Age"])
# %%
# estimated salay bhi normal distribution hai
# par isko trainsformation kar sakte hai
sns.kdeplot(X["EstimatedSalary"])
# %%
# ek bar QQ plot bnhi ry karte hai
# hume phele train test split kar lena chaiye
# %%
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
X_train
# %%
import scipy.stats as stats

stats.probplot(X["Age"], dist="norm", plot=plt)
plt.show()
# %%
stats.probplot(X["EstimatedSalary"], dist="norm", plot=plt)
plt.show()
# %%
# data kafi acha hai normalised hai toh hum chek karenge hi model ki accuracy kharab hoiti hai ki nahi transformation se
#

# %%
# without transformation both logisstic regression and decision tree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

clf = LogisticRegression()
clf2 = DecisionTreeClassifier()
# %%
clf.fit(X_train, Y_train)
clf2.fit(X_train, Y_train)

y_pred = clf.predict(X_test)
y_pred2 = clf2.predict(X_test)

print("accuracy of LR", accuracy_score(Y_test, y_pred))
print("accuracy of DT", accuracy_score(Y_test, y_pred2))
# %%
# dono se 81 percent
# lets se ab transformation ke badh
# phele log transformation karte hai dono coluumn par fir ek ek par
# then function banake dheakte hai
# %%
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import FunctionTransformer

trf = FunctionTransformer(np.log1p)
# %%
X_train_transformed = trf.fit_transform(X_train)
X_test_transformed = trf.transform(X_test)
# %%
clf = LogisticRegression()
clf2 = DecisionTreeClassifier()
clf.fit(X_train_transformed, Y_train)
clf2.fit(X_train_transformed, Y_train)

y_pred = clf.predict(X_test)
y_pred2 = clf2.predict(X_test)

print("accuracy of LR", accuracy_score(Y_test, y_pred))
print("accuracy of DT", accuracy_score(Y_test, y_pred2))

# %%
trf = FunctionTransformer(np.log1p)
# %%
X_train_transformed = trf.fit_transform(X_train[["Age"]])
X_test_transformed = trf.transform(X_test[["Age"]])
# %%
clf = LogisticRegression()
clf2 = DecisionTreeClassifier()

clf.fit(X_train_transformed, Y_train)
clf2.fit(X_train_transformed, Y_train)

y_pred = clf.predict(X_test_transformed)
y_pred2 = clf.predict(X_test_transformed)

print("accuracy of LR", accuracy_score(Y_test, y_pred))
print("accuracy of DT", accuracy_score(Y_test, y_pred))

# %%
# 81 se 86 ho gaya age ko kar ke log transform
# functionbanate hai


# %%
def apply_transform(transform):
    X = df.iloc[:, 0:2]
    Y = df.iloc[:, 2]

    trf = ColumnTransformer(
        [("log", FunctionTransformer(transform), ["Age"])], remainder="passthrough"
    )

    X_trans = trf.fit_transform(X)

    clf = LogisticRegression()

    print(
        "acc",
        np.mean(cross_val_score(clf, X_trans, Y, scoring="accuracy", cv=5)),
    )


# %%
apply_transform(lambda x: x**1 / 2)
