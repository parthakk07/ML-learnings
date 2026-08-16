# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# %%
df = pd.read_csv("/home/parthak/Videos/autoYT/wine_data.csv", usecols=[0, 1, 2])
df.columns = ["Class label", "Alcohol", "Malic acid"]
# %%
df.sample()

# %%
sns.kdeplot(df["Alcohol"])
sns.kdeplot(df["Malic acid"])
# minmax normalization
# %%
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(
    df.drop("Class label", axis=1), df["Class label"], test_size=0.3, random_state=0
)
# %%
X_train.shape, X_test.shape
# %%
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %%
X_test_scaled
# %%
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_train.columns)
# %%
X_train_scaled
# %%
sns.scatterplot(x=X_train["Alcohol"], y=X_train["Malic acid"])
# %%
sns.scatterplot(x=X_train_scaled["Alcohol"], y=X_train_scaled["Malic acid"])
# %%
# no data change
sns.kdeplot(X_train)
# %%
sns.kdeplot(X_train_scaled)
# %%
