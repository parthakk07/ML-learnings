# %%
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.linear_model import SGDRegressor

# %%
X, y = load_diabetes(return_X_y=True)
# %%
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(
    X, y, test_size=0.2, random_state=44
)
X_train
# %%
reg = SGDRegressor(max_iter=100, learning_rate="constant", eta0=0.01)
# %%
reg.fit(X_train, Y_train)
# %%
y_pred = reg.predict(X_test)
# %%
from sklearn.metrics import r2_score

r2_score(Y_test, y_pred)
# %%


# mini batch learning
sgd = SGDRegressor(learning_rate="constant", eta0=0.1)

# %%
batch_size = 35
import random

for i in range(100):
    idx = random.sample(range(X_train.shape[0]), batch_size)
    sgd.partial_fit(X_train[idx], Y_train[idx])
# %%
sgd.coef_
# %%
sgd.intercept_
# %%
y_pred = sgd.predict(X_test)
# %%
r2_score(Y_test, y_pred)
# %%
