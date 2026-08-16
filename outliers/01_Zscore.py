# %%
import numpy as np
import pandas as pd
import seaborn as sns

# %%
df = pd.read_csv("/home/parthak/Videos/autoYT/placement.csv")
df.sample(5)
# %%
sns.kdeplot(df["cgpa"])
# %%
sns.kdeplot(df["placement_exam_marks"])
# %%
# cgpa have normal distribution==> zscore
# exammarks naot having normal distribution
print("mean", df["cgpa"].mean())
print("std", df["cgpa"].std())
print("min", df["cgpa"].min())
print("max", df["cgpa"].max())
# %%
# finding outlers
print("highest", df["cgpa"].mean() + 3 * (df["cgpa"].std()))
print("lowest", df["cgpa"].mean() - 3 * (df["cgpa"].std()))
# %%
df[(df["cgpa"] > 8.80) | (df["cgpa"] < 5.11)]
# %%
# 5 outliers we can trim them
new_df = df[
    (df["cgpa"] < (df["cgpa"].mean() + 3 * (df["cgpa"].std())))
    & (df["cgpa"] > (df["cgpa"].mean() - 3 * (df["cgpa"].std())))
]
new_df
# %%
# trimmed the outliers
# now we can also do the z score
df["cgpa_Zscore"] = (df["cgpa"] - df["cgpa"].mean()) / df["cgpa"].std()
df.sample(3)
# %%
new_df = df[(df["cgpa_Zscore"] > -3) & (df["cgpa_Zscore"] < 3)]
new_df
# %%
# capping
higher = df["cgpa"].mean() + 3 * df["cgpa"].std()
lower = df["cgpa"].mean() - 3 * df["cgpa"].std()
# %%
df["cgpa"] = np.where(
    df["cgpa"] > higher, higher, np.where(df["cgpa"] < lower, lower, df["cgpa"])
)
df
# %%
df["cgpa"].describe()
# %%
sns.kdeplot(df["cgpa"])
