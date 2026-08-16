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
df["placement_exam_marks"].skew()
# %%
sns.boxplot(x=df["placement_exam_marks"])
# %%
df["placement_exam_marks"].describe()
# %%
# 25%ile = 17
# 75%ile = 44
percentile25 = df["placement_exam_marks"].quantile(0.25)
percentile75 = df["placement_exam_marks"].quantile(0.75)
# %%
iqr = percentile75 - percentile25
# %%
upperlimit = percentile75 + 1.5 * iqr
lowerlimit = percentile25 - 1.5 * iqr
# %%
# finding the outliers
df[
    (df["placement_exam_marks"] > upperlimit)
    | (df["placement_exam_marks"] < lowerlimit)
]
# %%
new_df = df[
    (df["placement_exam_marks"] < upperlimit)
    & (df["placement_exam_marks"] > lowerlimit)
]
new_df
# %%
sns.boxplot(x=df["placement_exam_marks"])
# %%
sns.boxplot(x=new_df["placement_exam_marks"])
# %%
new_cap_df = df.copy()
# %%
# capping
new_cap_df["placement_exam_marks"] = np.where(
    df["placement_exam_marks"] > upperlimit,
    upperlimit,
    np.where(
        df["placement_exam_marks"] < lowerlimit, lowerlimit, df["placement_exam_marks"]
    ),
)
# %%
new_cap_df
# %%
sns.boxplot(new_cap_df["placement_exam_marks"])
# %%
sns.kdeplot(new_cap_df["placement_exam_marks"])
