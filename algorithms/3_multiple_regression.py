# %%
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# %%
from sklearn.datasets import make_regression

x, y = make_regression(
    n_samples=100, n_features=2, n_informative=2, n_targets=1, noise=50
)
# %%
df = pd.DataFrame({"feature1": X[:, 0], "feature2": X[:, 1], "target": y})
# %%
df.sample(5)
# %%
fig = px.scatter_3d(df, x="feature1", y="feature2", z="target")

fig.show()
# %%
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=3)
# %%
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, Y_train)
# %%
y_pred = lr.predict(X_test)
# %%
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("MAE", mean_absolute_error(Y_test, y_pred))
print("MSE", mean_squared_error(Y_test, y_pred))
print("R2 score", r2_score(Y_test, y_pred))
# %%
lr.coef_
# %%
lr.intercept_
# %%
