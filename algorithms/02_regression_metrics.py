# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# %%
df = pd.read_csv("/home/parthak/Videos/autoYT/placement2.csv")
df.sample(4)
# %%
x = df.drop(["package"], axis=1)
y = df["package"]
# %%
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=3)
X_train
# %%
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, Y_train)
y_pred = lr.predict(X_test)
# %%
import seaborn as sns

sns.scatterplot(x=df["cgpa"], y=df["package"])
# %%
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("mae", mean_absolute_error(Y_test, y_pred))
print("mse", mean_squared_error(Y_test, y_pred))
print("R2", r2_score(Y_test, y_pred))

r2 = r2_score(Y_test, y_pred)
print("Adjusted R2 score", 1 - ((1 - r2) * (40 - 1) / (40 - 1 - 1)))


# %%
r2 = r2_score(y_pred, Y_test)
print("Adjusted R2 score", 1 - ((1 - r2) * (40 - 1) / (40 - 1 - 1)))
# %%
new_df1 = df.copy()
new_df1["random_feature"] = np.random.random(200)

new_df1 = new_df1[["cgpa", "random_feature", "package"]]
new_df1.head()
# %%
plt.scatter(new_df1["random_feature"], new_df1["package"])
plt.xlabel("random_feature")
plt.ylabel("Package(in lpa)")
# %%
X = new_df1.iloc[:, 0:2]
y = new_df1.iloc[:, -1]
# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
# %%
lr = LinearRegression()
lr.fit(X_train, Y_train)
y_pred2 = lr.predict(X_test)
# %%
print("R2 score", r2_score(y_test, y_pred2))
r2 = r2_score(y_test, y_pred2)
# %%
1 - ((1 - r2) * (40 - 1) / (40 - 1 - 2))
# %%
# after adding a random column the r2 score become negative
#
