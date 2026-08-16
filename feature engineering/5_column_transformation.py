# %%
import numpy as np
import pandas as pd

# %%
df = pd.read_csv("/home/parthak/Videos/autoYT/covid_toy.csv")
df.sample(5)

# %%
df["cough"].value_counts()

# %%
df["city"].value_counts()

# %%
df["gender"].value_counts()

# %%
df.isnull().sum()
# %%
# fever have missing values == fill the data or remove data
# 4 states are there kolkata and delhi and mumbai and banglore ==one hot encoding
# gender male and female == one hot encoding
# cough typoes mild and strong == ordianl encoding
# has_covid == label encoding

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(
    df.drop("has_covid", axis=1), df["has_covid"], test_size=0.3, random_state=0
)
X_train.sample(5)

# %%
from sklearn.preprocessing import OrdinalEncoder

oe = OrdinalEncoder(categories=[["Mild", "Strong"]])
X_train_oe = oe.fit_transform(X_train[["cough"]])
X_test_oe = oe.fit_transform(X_test[["cough"]])

# %%
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(drop="first")
X_train_ohe = ohe.fit_transform(X_train[["gender", "city"]]).toarray()
X_test_ohe = ohe.fit_transform(X_test[["gender", "city"]]).toarray()
X_train_ohe

# %%
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
Y_train_le = le.fit_transform(Y_train)
Y_test_le = le.fit_transform(Y_test)
Y_train_le

# %%
# missing data in fever column
from sklearn.impute import SimpleImputer

si = SimpleImputer()
X_train_si = si.fit_transform(X_train[["fever"]])
X_test_si = si.fit_transform(X_test[["fever"]])
X_train_si

# %%
X_train_age = X_train[["age"]].values
X_test_age = X_test[["age"]].values
# %%
# combine all
X_train_transformed = np.concatenate(
    (X_train_age, X_train_si, X_train_oe, X_train_ohe), axis=1
)
X_test_transformed = np.concatenate(
    (X_test_age, X_test_si, X_test_oe, X_test_ohe), axis=1
)

# %%
pd.DataFrame(X_train_transformed)
# %%


# mentos zindigii
#
from sklearn.compose import ColumnTransformer

transformer = ColumnTransformer(
    transformers=[
        ("tnf1", SimpleImputer(), ["fever"]),
        ("tnf2", OrdinalEncoder(categories=[["Mild", "Strong"]]), ["cough"]),
        ("tnf3", OneHotEncoder(sparse_output=False, drop="first"), ["gender", "city"]),
    ],
    remainder="passthrough",
)

# %%
X_train_transformed = transformer.fit_transform(X_train)
X_test_transformed = transformer.transform(X_test)

# %%
X_train_transformed = pd.DataFrame(
    X_train_transformed, columns=transformer.get_feature_names_out()
)
X_test_transformed = pd.DataFrame(
    X_test_transformed, columns=transformer.get_feature_names_out()
)

# %%
X_train_transformed
