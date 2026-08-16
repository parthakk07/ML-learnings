# %%
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# %%
X = 6 * np.random.rand(200, 1) - 3
y = 0.8 * X**2 + 0.9 * X + 2 + np.random.randn(200, 1)

# y = 0.8x^2 + 0.9x + 2
# %%
X_train, X_test, Y_train, Y_test = train_test_split(
    X, y, test_size=0.2, random_state=44
)
X_train
# %%
lr = LinearRegression()
lr.fit(X_train, Y_train)
y_pred = lr.predict(X_test)
# %%
from sklearn.metrics import r2_score

# %%
r2_score(Y_test, y_pred)
# %%
# 38%

# %%
poly = PolynomialFeatures(degree=2)
# %%
X_train_trans = poly.fit_transform(X_train)
X_test_trans = poly.transform(X_test)
# %%
X_train[0]
# %%
X_train_trans[0]
# %%
lr.fit(X_train_trans, Y_train)
y_pred = lr.predict(X_test_trans)
# %%
r2_score(Y_test, y_pred)
# %%
print(lr.coef_)
print(lr.intercept_)

# %%
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline


def polynomial_regression(degree):
    X_new = np.linspace(-3, 3, 100).reshape(100, 1)
    X_new_poly = poly.transform(X_new)

    polybig_features = PolynomialFeatures(degree=degree, include_bias=False)
    std_scaler = StandardScaler()
    lin_reg = LinearRegression()
    polynomial_regression = Pipeline(
        [
            ("poly_features", polybig_features),
            ("std_scaler", std_scaler),
            ("lin_reg", lin_reg),
        ]
    )
    polynomial_regression.fit(X, y)
    y_newbig = polynomial_regression.predict(X_new)
    plt.plot(X_new, y_newbig, "r", label="Degree " + str(degree), linewidth=2)

    plt.plot(X_train, Y_train, "b.", linewidth=3)
    plt.plot(X_test, Y_test, "g.", linewidth=3)
    plt.legend(loc="upper left")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.axis([-3, 3, 0, 10])
    plt.show()


# %%
polynomial_regression(4)
# %%
