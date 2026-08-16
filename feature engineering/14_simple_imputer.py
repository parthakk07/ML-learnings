# %%
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# %%
df = pd.read_csv("/home/parthak/Videos/autoYT/titanic_toy.csv")
df.sample(5)
# %%
df.info()
# %%
df.isnull().mean()
# %%
X = df.drop(["Survived"], axis=1)
Y = df["Survived"]
X
# %%
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.3, random_state=42
)
X_train
# %%
X_train.shape
# %%
X_test.shape
# %%
X_train.isnull().mean()
# %%
mean_age = X_train["Age"].mean()
median_age = X_train["Age"].median()

mean_fare = X_train["Fare"].mean()
median_fare = X_train["Fare"].median()
# %%
X_train["mean_age"] = X_train["Age"].fillna(mean_age)
X_train["median_age"] = X_train["Age"].fillna(median_age)

X_train["mean_fare"] = X_train["Fare"].fillna(mean_fare)
X_train["median_fare"] = X_train["Fare"].fillna(median_fare)

X_train
# %%
# chack variance for the mean and median columns with orignal columns
import seaborn as sns

sns.kdeplot(X_train["Age"], color="black")

sns.kdeplot(X_train["mean_age"], color="green")
sns.kdeplot(X_train["median_age"], color="yellow")
# here distribution changed a lot thsts a red flag also affect the model
# also there were 20 % dat missiing
# %%

sns.kdeplot(X_train["Fare"], color="black")

sns.kdeplot(X_train["mean_fare"], color="green")
sns.kdeplot(X_train["median_fare"], color="yellow")
# alsomst same distribution so here filling with mean or median is good decison
# %%
# checking vaariance
print("orignal varaiance ", X_train["Age"].var())
print("mean imputer varaiance ", X_train["mean_age"].var())
print("median imputere varaiance ", X_train["median_age"].var())

print("orignal varaiance ", X_train["Fare"].var())
print("mean imputer varaiance ", X_train["mean_fare"].var())
print("median imputere varaiance ", X_train["median_fare"].var())

# %%
# age columns have lot of change in the variance is a red flag

# checking covariance
X_train.cov()
# %%
X_train.corr()
# %%
X_train[["Age", "mean_age", "median_age"]].boxplot()
# in box plot many outliers added IQR shrink
# %%
X_train[["Fare", "mean_fare", "median_fare"]].boxplot()
# same as before mean and median imputation is  good for fare column
#  not for age column also age have lot of data missing

# %%
# using sklearn
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=43
)
X_train
# %%
impute1 = SimpleImputer(strategy="median")
impute2 = SimpleImputer(strategy="mean")

# %%
trf = ColumnTransformer(
    [("impute1", impute1, ["Age"]), ("impute2", impute2, ["Fare"])],
    remainder="passthrough",
)
# %%
X_train_transformed = trf.fit_transform(X_train)
X_test_transformed = trf.transform(X_test)
# %%
pd.DataFrame(X_train_transformed)
# %%
X_test_transformed
# %%
# arbitary value iimpute
# using sklearn
impute1 = SimpleImputer(strategy="constant", fill_value=99)
impute2 = SimpleImputer(strategy="constant", fill_value=999)

# %%
trf = ColumnTransformer(
    [("impute1", impute1, ["Age"]), ("impute2", impute2, ["Fare"])],
    remainder="passthrough",
)
# %%
trf.fit(X_train)
# %%
trf.named_transformers_["impute1"].statistics_
# %%
X_train = trf.transform(X_train)
X_test = trf.transform(X_test)
# %%
X_train
