# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# %%
df = pd.read_csv("/home/parthak/Videos/autoYT/titanic/train.csv")

# %%
df.sample(5)

# categorical data

# %%
sns.countplot(x=df["Survived"])
# %%
sns.countplot(x=df["Pclass"])
# %%
sns.countplot(x=df["Embarked"])

# %%
sns.countplot(x=df["Sex"])
# %%
df["Survived"].value_counts().plot(kind="pie", autopct="%.2f")
# numericcal data

# %%
# histogram
plt.hist(df["Fare"])
# %%
plt.hist(df["Age"], bins=5)
# %%
# distploot
sns.distplot(df["Age"])
# %%
sns.histplot(x=df["Age"])
sns.displot(x=df["Age"])
# %%
#  distplot old new is dispplot and histplot
# kde- kernal density estimation tells distribution of data
sns.histplot(df["Age"], kde=True)
# %%
# boxplot gives 5 number summary
sns.boxplot(x=df["Age"])
# %%
print(df["Age"].min())
print(df["Age"].max())
print(df["Age"].mean())
# %%
