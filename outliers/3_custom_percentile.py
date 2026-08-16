# %%
import numpy as np
import pandas as pd
import seaborn as sns

# %%
df = pd.read_csv("/home/parthak/Videos/autoYT/weight-height.csv")
df.sample(5)
# %%
sns.kdeplot(df["Height"])
# %%
sns.boxplot(x=df["Height"])
# %%
sns.kdeplot(df["Weight"])
# %%
sns.boxplot(x=df["Weight"])
# %%
upperr_limit = df["Height"].quantile(0.99)
lower_limit = df["Height"].quantile(0.01)
# %%
df[(df["Height"] >= upperr_limit) | (df["Height"] <= lower_limit)]
# %%
new_df = df[(df["Height"] <= upperr_limit) & (df["Height"] >= lower_limit)]
new_df
# %%
sns.boxplot(x=new_df["Height"])
# %%
# capping - winserazaation aka
df["Height"] = np.where(
    df["Height"] >= upperr_limit,
    upperr_limit,
    np.where(df["Height"] <= lower_limit, lower_limit, df["Height"]),
)
# %%
df["Height"].describe()
# %%
sns.boxplot(df["Height"])
# %%
