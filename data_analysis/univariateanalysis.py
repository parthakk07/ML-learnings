 # %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# %%
df = pd.read_csv("/home/parthak/Videos/autoYT/titanic/train.csv")
df.sample(5)

# %%
# sns.countplot(df["Survived"])
# sns.countplot(df["Embarked"])
df["Survived"].value_counts().plot(kind="bar")
# %%
df["Survived"].value_counts().plot(kind="pie", autopct="%.2f")

# %%
plt.hist(df["Age"], bins=10)

# %%
sns.distplot(df["Age"])

# %%
