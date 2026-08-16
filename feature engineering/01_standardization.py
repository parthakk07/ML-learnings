# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# %%
df = pd.read_csv("/home/parthak/Videos/autoYT/Social_Network_Ads.csv")
df.sample(5)

# %%
df = df.iloc[:, 2:]

# %%
df.sample(5)

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(
    df.drop("Purchased", axis=1), df["Purchased"], test_size=0.3, random_state=0
)

# %%
X_train.sample(5)
# %%
# standardization
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
# fit the scaler to the train set, it will learn the parameters
scaler.fit(X_train)

# transform train and test sets
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
# scalled are in np array convert it into dataframe
# %%
X_train_scaled
# %%
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)

X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
# %%
X_train_scaled
# %%
sns.scatterplot(x=X_train["Age"], y=X_train["EstimatedSalary"])
# %%
sns.scatterplot(x=X_train_scaled["Age"], y=X_train_scaled["EstimatedSalary"])
# %%
sns.kdeplot(X_train["Age"])
sns.kdeplot(X_train["EstimatedSalary"])
# %%
# only scale changed no data changed
sns.kdeplot(X_train_scaled["Age"])
sns.kdeplot(X_train_scaled["EstimatedSalary"])
# %%
