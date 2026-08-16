# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# %%
from sklearn.datasets import load_diabetes

data = load_diabetes()
# %%
X = data.data
y = data.target
# %%
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(
    X, y, test_size=0.2, random_state=44
)
X_train
# %%
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, Y_train)
y_pred = lr.predict(X_test)
# %%
from sklearn.metrics import r2_score

r2_score(Y_test, y_pred)
# %%
# 52 r2 socre
from sklearn.linear_model import Ridge

r = Ridge(alpha=50)
r.fit(X_train, Y_train)
y_pred2 = r.predict(X_test)
r2_score(Y_test, y_pred2)
# %%
m = 100
x1 = 5 * np.random.rand(m, 1) - 2
x2 = 0.7 * x1**2 - 2 * x1 + 3 + np.random.randn(m, 1)

plt.scatter(x1, x2)
plt.show()
# %%
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures


def get_preds_ridge(x1, x2, alpha):
    model = Pipeline(
        [("poly_feats", PolynomialFeatures(degree=16)), ("ridge", Ridge(alpha=alpha))]
    )
    model.fit(x1, x2)
    return model.predict(x1)


alphas = [0, 20, 200]
cs = ["r", "g", "b"]

plt.figure(figsize=(10, 6))
plt.plot(x1, x2, "b+", label="Datapoints")

for alpha, c in zip(alphas, cs):
    preds = get_preds_ridge(x1, x2, alpha)
    # Plot
    plt.plot(
        sorted(x1[:, 0]),
        preds[np.argsort(x1[:, 0])],
        c,
        label="Alpha: {}".format(alpha),
    )

plt.legend()
plt.show()
# %%
