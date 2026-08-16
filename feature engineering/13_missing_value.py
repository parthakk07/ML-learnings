# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# %%
import seaborn as sns

# %%
df = pd.read_csv("/home/parthak/Videos/autoYT/data_science_job.csv")
df.sample(5)

# %%
df.isnull().sum()
# %%
df.shape
# %%
df.isnull().mean() * 100
# %%
# 5 % se kam walo par cca apply karenge
# baki columns ko hatana better arahega kyuki data kam bachega fir

# %%
cols = [
    x for x in df.columns if df[x].isnull().mean() <= 0.05 and df[x].isnull().mean() > 0
]
cols
# %%
df[cols].sample(5)
# %%
len(df[cols].dropna()) / len(df)

# %%
new_df = df[cols].dropna()
new_df.shape, df.shape
# %%
new_df.isnull().sum()
# %%
sns.histplot(df["city_development_index"])
sns.histplot(new_df["city_development_index"])

# %%
sns.histplot(df["experience"])
sns.histplot(new_df["experience"], color="red")
# %%
sns.histplot(df["training_hours"])
sns.histplot(new_df["training_hours"], color="red")
# %%
sns.kdeplot(df["training_hours"])
sns.kdeplot(new_df["training_hours"], color="red")
# %%
df["education_level"].value_counts()
# %%
# almost same hai sab ka histplot and pdf/kde new df and old df so apply cca is good hjere nothing bothering
temp = pd.concat(
    [
        # percentage of observations per category, original data
        df["enrolled_university"].value_counts() / len(df),
        # percentage of observations per category, cca data
        new_df["enrolled_university"].value_counts() / len(new_df),
    ],
    axis=1,
)

# add column names
temp.columns = ["original", "cca"]

temp


# %%
temp = pd.concat(
    [
        # percentage of observations per category, original data
        df["education_level"].value_counts() / len(df),
        # percentage of observations per category, cca data
        new_df["education_level"].value_counts() / len(new_df),
    ],
    axis=1,
)

# add column names
temp.columns = ["original", "cca"]

temp
# %%
# ratio is also similatr to orignal df
