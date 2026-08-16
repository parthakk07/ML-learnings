# %%
import numpy as np
import pandas as pd
from pandas.core.common import random_state

# %%
df = pd.read_csv("/home/parthak/Videos/autoYT/cars.csv")
df.sample(5)
# %%
df["brand"].value_counts()
# %%
df.nunique()
# %%
# using pandas
pd.get_dummies(df, columns=["fuel", "owner"])

# %%
# pandas not used coz it is randomized
pd.get_dummies(df, columns=["fuel", "owner"], drop_first=True)
# %%
df.sample(5)
# %%
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(
    df.iloc[:, 0:4], df.iloc[:, -1], test_size=0.3, random_state=0
)
# %%
X_train.shape
# %%
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder()
X_train_new = ohe.fit_transform(X_train[["fuel", "owner"]]).toarray()
X_test_new = ohe.fit_transform(X_test[["fuel", "owner"]]).toarray()
# %%
X_train_new

# %%
np.hstack((X_train[["brand", "km_driven"]].values, X_train_new))
# %%
X_temp = pd.DataFrame(np.hstack((X_train[["brand", "km_driven"]].values, X_train_new)))

# %%
X_temp.shape
# %%
# multicolinearity na ho toh uske liye
ohe = OneHotEncoder(drop="fisrt", dtype=np.int32)

# %%
# one hot encoding in column having nimber of categories very high
counts = df["brand"].value_counts()
counts
# %%
counts.nunique()
threshold = 100
# %%
repl = counts[counts <= threshold].index
repl

# %%
pd.get_dummies(df["brand"].replace(repl, "uncommon")).sample(5)

# %%
