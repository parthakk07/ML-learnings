# %%
import numpy as np
import pandas as pd

# %%
df = pd.read_csv("/home/parthak/Videos/autoYT/customer.csv")
df.sample(5)
# %%
df = df.iloc[:, 2:]
df.sample(5)
# %%
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(
    df.drop("purchased", axis=1), df["purchased"], test_size=0.3, random_state=0
)
X_train.sample(5)
# %%
Y_train.sample(5)
# %%
from sklearn.preprocessing import OrdinalEncoder

oe = OrdinalEncoder(categories=[["Poor", "Average", "Good"], ["School", "UG", "PG"]])
oe.fit(X_train)
X_train_oe = oe.transform(X_train)
X_test_oe = oe.transform(X_test)

# %%
X_train
# %%
X_train_oe = pd.DataFrame(X_train_oe, columns=X_train.columns)
X_test_oe = pd.DataFrame(X_test_oe, columns=X_train.columns)
# %%
Y_train
# %%
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le.fit(Y_train)
Y_train_le = le.transform(Y_train)
Y_test_le = le.transform(Y_test)

# %%
Y_test_le
# %%
