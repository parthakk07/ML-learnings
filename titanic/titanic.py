# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# %%
df = pd.read_csv("train.csv")
df.head()
# %%
df.info()
# %%
df["Age"].hist()
# %%
df["Age"].plot()
# %%

plt.scatter(df["Age"], df["Pclass"])


#  %%
pd.crosstab(df["Survived"], df["Pclass"]).plot()
