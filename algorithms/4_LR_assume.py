# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# %%
df = pd.read_csv("/home/parthak/Videos/autoYT/data.csv")
df.sample(5)
# %%
X = df.iloc[:, 0:3].values
y = df.iloc[:, -1].values
# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=33
)
X_train
# %%
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
# %%
#  residual
residual = y_test - y_pred
# %%
#  1 linerar relationship
fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(12, 2.5))

ax1.scatter(df["feature1"], df["target"])
ax1.set_title("Feature1")
ax2.scatter(df["feature2"], df["target"])
ax2.set_title("Feature2")
ax3.scatter(df["feature3"], df["target"])
ax3.set_title("Feature3")

plt.show()
# %%
# 2. Multicollinearity
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = []

for i in range(X_train.shape[1]):
    vif.append(variance_inflation_factor(X_train, i))
# %%
pd.DataFrame({"vif": vif}, index=df.columns[0:3]).T
# %%
# Another Technique
sns.heatmap(df.iloc[:, 0:3].corr(), annot=True)
# %%
# 3. Normality of Residual
sns.kdeplot(residual)
# %%
# QQ Plot
import scipy as sp

fig, ax = plt.subplots(figsize=(6, 4))
sp.stats.probplot(residual, plot=ax, fit=True)

plt.show()
# %%
# 4. Homoscedasticity
sns.scatterplot(residual)
# %%
# 5. Autocorrelation of Residuals
plt.plot(residual)

# %%
