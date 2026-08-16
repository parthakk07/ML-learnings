# %%
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer

# %%
df = pd.read_csv(
    "/home/parthak/Videos/autoYT/train.csv",
    usecols=["GarageQual", "FireplaceQu", "SalePrice"],
)
df.sample(3)
# %%
df.isnull().mean()
# %%
# 47 percent data missing in the  fireplaceQu column == replace with "missing"
# garageQual have 5 % data missing so we can use most freequent here
# phele dono par mode lagate hai fir dono par missing
# %%
sns.histplot(df["GarageQual"])
# %%
df["GarageQual"].mode()
# %%

import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)

df[df["GarageQual"] == "TA"]["SalePrice"].plot(kind="kde", ax=ax)

df[df["GarageQual"].isnull()]["SalePrice"].plot(kind="kde", ax=ax, color="red")

lines, labels = ax.get_legend_handles_labels()
labels = ["Houses with TA", "Houses with NA"]
ax.legend(lines, labels, loc="best")

plt.title("GarageQual")

# %%
temp = df[df["GarageQual"] == "TA"]["SalePrice"]
# %%
df["GarageQual"].fillna("TA", inplace=True)
# %%
df["GarageQual"].value_counts().plot(kind="bar")
# %%
fig = plt.figure()
ax = fig.add_subplot(111)


temp.plot(kind="kde", ax=ax)

# distribution of the variable after imputation
df[df["GarageQual"] == "TA"]["SalePrice"].plot(kind="kde", ax=ax, color="red")

lines, labels = ax.get_legend_handles_labels()
labels = ["Original variable", "Imputed variable"]
ax.legend(lines, labels, loc="best")

# add title
plt.title("GarageQual")

# %%
df["FireplaceQu"].value_counts().plot(kind="bar")
# %%
df["FireplaceQu"].mode()
# %%
fig = plt.figure()
ax = fig.add_subplot(111)

df[df["FireplaceQu"] == "Gd"]["SalePrice"].plot(kind="kde", ax=ax)

df[df["FireplaceQu"].isnull()]["SalePrice"].plot(kind="kde", ax=ax, color="red")

lines, labels = ax.get_legend_handles_labels()
labels = ["Houses with Gd", "Houses with NA"]
ax.legend(lines, labels, loc="best")

plt.title("FireplaceQu")
# %%
temp = df[df["FireplaceQu"] == "Gd"]["SalePrice"]
# %%
df["FireplaceQu"].fillna(df["FireplaceQu"].mode(), inplace=True)
# %%
df["FireplaceQu"].value_counts().plot(kind="bar")
# %%
fig = plt.figure()
ax = fig.add_subplot(111)


temp.plot(kind="kde", ax=ax)

# distribution of the variable after imputation
df[df["FireplaceQu"] == "Gd"]["SalePrice"].plot(kind="kde", ax=ax, color="red")

lines, labels = ax.get_legend_handles_labels()
labels = ["Original variable", "Imputed variable"]
ax.legend(lines, labels, loc="best")

# add title
plt.title("FireplaceQu")

# %%
# using sklearns
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(
    df.drop(["SalePrice"], axis=1), df["SalePrice"], test_size=0.2, random_state=44
)
X_train
# %%
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="most_frequent")

# %%
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)
# %%
X_train
# %%
imputer.statistics_
# %%
# making missing categgory
# %%
df = pd.read_csv(
    "/home/parthak/Videos/autoYT/train.csv",
    usecols=["GarageQual", "FireplaceQu", "SalePrice"],
)
df.sample(3)
# %%
df["FireplaceQu"].fillna("missing", inplace=True)
# %%
sns.histplot(df["FireplaceQu"])
# %%
df["GarageQual"].fillna("Missing", inplace=True)
# %%
df["GarageQual"].value_counts().sort_values(ascending=False).plot.bar()
plt.xlabel("GarageQual")
plt.ylabel("Number of houses")
# %%
# usnig sklearns
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="constant", fill_value="missing")
# %%
X_train, X_test, Y_train, Y_test = train_test_split(
    df.drop(["SalePrice"], axis=1), df["SalePrice"], test_size=0.3
)
X_train
# %%
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)
X_train
# %%
imputer.statistics_
