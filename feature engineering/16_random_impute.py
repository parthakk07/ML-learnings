# %%
import numpy as np
import pandas as pd
from matplotlib.tri import TrapezoidMapTriFinder

# %%
df = pd.read_csv(
    "/home/parthak/Videos/autoYT/titanic/train.csv", usecols=["Age", "Fare", "Survived"]
)
df.sample(4)
# %%
df.isnull().mean()
# %%
X = df.drop(["Survived"], axis=1)
Y = df["Survived"]
# %%
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
X_train
# %%
X_train["Age_imputed"] = X_train["Age"]
X_test["Age_imputed"] = X_test["Age"]
# %%
X_train.head()
# %%
# this code is replacing the age_imputed data where nan is existed with random numbers form the age  data
X_train["Age_imputed"][X_train["Age_imputed"].isnull()] = (
    X_train["Age"].dropna().sample(X_train["Age"].isnull().sum()).values
)
X_test["Age_imputed"][X_test["Age_imputed"].isnull()] = (
    X_train["Age"].dropna().sample(X_test["Age"].isnull().sum()).values
)
# %%
X_train.isnull().sum()
# %%
X_train["Age"].dropna().sample(X_train["Age"].isnull().sum()).values
# %%
import seaborn as sns

sns.kdeplot(X_train["Age"])
sns.kdeplot(X_train["Age_imputed"])
# %%
print("orignal variancce", X_train["Age"].var())
print("imputed variancce", X_train["Age_imputed"].var())

# %%
X_train[["Fare", "Age", "Age_imputed"]].cov()
# %%
X_train[["Age", "Age_imputed"]].boxplot()
# %%
# for same values no different output
sampled_value = X_train["Age"].dropna().sample(1, random_state=int(observation["Fare"]))

# %%
# for catgotical
df = pd.read_csv(
    "/home/parthak/Videos/autoYT/train.csv",
    usecols=["GarageQual", "FireplaceQu", "SalePrice"],
)
df.sample(3)
# %%
x = df
y = df["SalePrice"]
# %%
X_train, X_test, Y_train, Y_test = train_test_split(
    x, y, test_size=0.2, random_state=33
)
# %%
X_train["GarageQual_imputed"] = X_train["GarageQual"]
X_test["GarageQual_imputed"] = X_test["GarageQual"]

X_train["FireplaceQu_imputed"] = X_train["FireplaceQu"]
X_test["FireplaceQu_imputed"] = X_test["FireplaceQu"]

# %%
X_train.sample(5)
# %%
X_train["GarageQual_imputed"][X_train["GarageQual_imputed"].isnull()] = (
    X_train["GarageQual"].dropna().sample(X_train["GarageQual"].isnull().sum()).values
)
X_test["GarageQual_imputed"][X_test["GarageQual_imputed"].isnull()] = (
    X_train["GarageQual"].dropna().sample(X_test["GarageQual"].isnull().sum()).values
)

X_train["FireplaceQu_imputed"][X_train["FireplaceQu_imputed"].isnull()] = (
    X_train["FireplaceQu"].dropna().sample(X_train["FireplaceQu"].isnull().sum()).values
)
X_test["FireplaceQu_imputed"][X_test["FireplaceQu_imputed"].isnull()] = (
    X_train["FireplaceQu"].dropna().sample(X_test["FireplaceQu"].isnull().sum()).values
)

# %%
X_train.sample(5)
# %%

temp = pd.concat(
    [
        X_train["GarageQual"].value_counts() / len(X_train["GarageQual"].dropna()),
        X_train["GarageQual_imputed"].value_counts() / len(X_train),
    ],
    axis=1,
)

temp.columns = ["original", "imputed"]

# %%
temp
# %%
temp = pd.concat(
    [
        X_train["FireplaceQu"].value_counts() / len(X_train["FireplaceQu"].dropna()),
        X_train["FireplaceQu_imputed"].value_counts() / len(df),
    ],
    axis=1,
)

temp.columns = ["original", "imputed"]

temp
# %%
for category in X_train["FireplaceQu"].dropna().unique():
    sns.kdeplot(X_train[X_train["FireplaceQu"] == category]["SalePrice"])
# %%
for category in X_train["FireplaceQu_imputed"].dropna().unique():
    sns.kdeplot(X_train[X_train["FireplaceQu_imputed"] == category]["SalePrice"])
# %%
# missing indicator
df = pd.read_csv(
    "/home/parthak/Videos/autoYT/titanic/train.csv", usecols=["Age", "Fare", "Survived"]
)
df.sample(5)
# %%
x = df.drop(["Survived"], axis=1)
y = df["Survived"]
# %%
X_train, X_test, Y_train, Y_test = train_test_split(
    x, y, test_size=0.2, random_state=33
)
X_train
# %%
# using pandas and missing class
from sklearn.impute import MissingIndicator, SimpleImputer

# first without missing column
si = SimpleImputer()
X_train_transformed = si.fit_transform(X_train)
X_test_transformed = si.transform(X_test)
# %%
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()
clf.fit(X_train_transformed, Y_train)
y_pred = clf.predict(X_test_transformed)
from sklearn.metrics import accuracy_score

accuracy_score(y_pred, Y_test)
# %%
mi = MissingIndicator()
mi.fit(X_train)
# %%
mi.features_
# %%
X_train_missing = mi.transform(X_train)
X_test_missing = mi.transform(X_test)
# %%
X_train["Age_NA"] = X_train_missing
X_test["Age_NA"] = X_test_missing
# %%
si = SimpleImputer()
X_train_trf2 = si.fit_transform(X_train)
X_test_trf2 = si.transform(X_test)
# %%
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()

clf.fit(X_train_trf2, Y_train)

y_pred = clf.predict(X_test_trf2)

from sklearn.metrics import accuracy_score

accuracy_score(Y_test, y_pred)
# %%
si = SimpleImputer(add_indicator=True)
X_train = si.fit_transform(X_train)
X_test = si.transform(X_test)
# %%
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()
clf.fit(X_train, Y_train)
y_pred = clf.predict(X_test)
from sklearn.metrics import accuracy_score

accuracy_score(Y_test, y_pred)
# %%
