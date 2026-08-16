# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.core.reshape.pivot import crosstab

# %%
titanic = pd.read_csv("/home/parthak/Videos/autoYT/titanic/train.csv")
titanic.sample(5)

# %%
tips = sns.load_dataset("tips")
tips.sample(5)

# %%
flight = sns.load_dataset("flights")
flight.sample(5)
# %%
iris = sns.load_dataset("iris")
iris.sample(5)
# %%
# numerical - numerical
sns.scatterplot(x=tips["total_bill"], y=tips["tip"], size=tips["size"])

# %%
sns.scatterplot(
    x=tips["total_bill"],
    y=tips["tip"],
    size=tips["size"],
    hue=tips["sex"],
    style=tips["smoker"],
)

# %%
# categorical an dnumerical
sns.barplot(x=titanic["Pclass"], y=titanic["Age"])

# %%
sns.barplot(x=titanic["Pclass"], y=titanic["Fare"], hue=titanic["Sex"])

# %%
# box plot
sns.boxplot(x=titanic["Sex"], y=titanic["Age"])

# %%
sns.boxplot(x=titanic["Sex"], y=titanic["Age"], hue=titanic["Survived"])

# %%
# displot
sns.displot(x=titanic[titanic["Survived"] == 0]["Age"], kde=True)
sns.displot(x=titanic[titanic["Survived"] == 1]["Age"], kde=True)
# %%
sns.displot(x=titanic["Age"], hue=titanic["Survived"], kde=True)

# %%
sns.displot(x=titanic["Age"], hue=titanic["Survived"], kind="kde", fill=True)

# %%
# categrocial - categorical
pd.crosstab(titanic["Pclass"], titanic["Survived"])

# %%
sns.heatmap(pd.crosstab(titanic["Pclass"], titanic["Survived"]))

# %%
titanic.groupby("Pclass")["Survived"].mean() * 100
# %%
titanic.groupby("Sex")["Survived"].mean() * 100

# %%
sns.clustermap(pd.crosstab(titanic["SibSp"], titanic["Survived"]))

# %%
# pairpot
sns.pairplot(iris, hue="species")

# %%
new = flight.groupby("year")["passengers"].sum().reset_index()
# %%
sns.lineplot(x=new["year"], y=new["passengers"])
# %%
flight.pivot_table(values="passengers", index="month", columns="year")
# %%
sns.heatmap(flight.pivot_table(values="passengers", index="month", columns="year"))
