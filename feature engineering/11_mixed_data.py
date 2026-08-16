# %%
import numpy as np
import pandas as pd

# %%
import seaborn as sns

# %%
df = pd.read_csv("/home/parthak/Videos/autoYT/titanic.csv")
df.sample(5)
# %%
df["number"].unique()
# %%
sns.histplot(df["number"])
# %%
# extract numerical part
df["number_numerical"] = pd.to_numeric(
    df["number"], errors="coerce", downcast="integer"
)
# %%
# extract categorical part
df["number_categorical"] = np.where(
    df["number_numerical"].isnull(), df["number"], np.nan
)

df.head()
# %%
df["Cabin"].unique()
# %%
df["Ticket"].unique()
# %%
# for cabin first is categorical and second part is numerical
#
df["cabin_num"] = df["Cabin"].str.extract("(\d+)")  # captures numerical part
df["cabin_cat"] = df["Cabin"].str[0]  # captures the first letter

df.head()
# %%
df["cabin_cat"].value_counts().plot(kind="bar")
# %%
# extract the last bit of ticket as number
df["ticket_num"] = df["Ticket"].apply(lambda s: s.split()[-1])
df["ticket_num"] = pd.to_numeric(df["ticket_num"], errors="coerce", downcast="integer")

# extract the first part of ticket as category
df["ticket_cat"] = df["Ticket"].apply(lambda s: s.split()[0])
df["ticket_cat"] = np.where(df["ticket_cat"].str.isdigit(), np.nan, df["ticket_cat"])

df.head(20)

# %%
