# %%

import numpy as np
from sklearn.datasets import make_regression

# %%
X, y = make_regression(
    n_samples=4, n_features=1, n_informative=1, n_targets=1, noise=80, random_state=13
)
# %%
import seaborn as sns

sns.scatterplot(X)

# %%
# from sklearn.model_selection import train_test_split

# X_train, X_test, Y_train, Y_test = train_test_split(
#     X, y, test_size=0.2, random_state=44
# )
# X_train

# %%
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X, y)
y_pred = lr.predict(X)

# %%
lr.coef_
# %%
lr.intercept_
# %%
import matplotlib.pyplot as plt

plt.scatter(X, y)
plt.plot(X, lr.predict(X), color="red")
# %%
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score

# %%
X, y = make_regression(
    n_samples=100, n_features=1, n_informative=1, n_targets=1, noise=20, random_state=13
)
# %%
plt.scatter(X, y)
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
lr.coef_
# %%
lr.intercept_


# %%
# m=27.83
# b=-1.8
# %%
class GDRegressor:
    def __init__(self, learning_rate, epochs):
        self.m = 100
        self.b = -120
        self.lr = learning_rate
        self.epochs = epochs

    def fit(self, X, y):
        # calcualte the b using GD
        for i in range(self.epochs):
            loss_slope_b = -2 * np.sum(y - self.m * X.ravel() - self.b)
            loss_slope_m = -2 * np.sum((y - self.m * X.ravel() - self.b) * X.ravel())

            self.b = self.b - (self.lr * loss_slope_b)
            self.m = self.m - (self.lr * loss_slope_m)
        print(self.m, self.b)

    def predict(self, X):
        return self.m * X + self.b


# %%
gr = GDRegressor(0.001, 50)
# %%
gr.fit(X_train, Y_train)
# %%
gr.predict(X_test)
# %%
r2_score(Y_test, gr.predict(X_test))
# %%
