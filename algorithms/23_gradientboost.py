# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.plotting import plot_decision_regions
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_circles
# %%
np.random.seed(42)
X, y = make_circles(n_samples=500, factor=0.1, noise=0.35, random_state=42)
# %%
plt.scatter(X[:,0],X[:,1],c=y)
# %%
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.2,random_state=42)
X_train
# %%
from sklearn.ensemble import GradientBoostingClassifier
gbs=GradientBoostingClassifier()
gbs.fit(X_train,Y_train)
y_pred=gbs.predict(X_test)
# %%
from sklearn.metrics import classification_report
classification_report(Y_test,y_pred)
# %%
from sklearn.metrics import accuracy_score
accuracy_score(Y_test,y_pred)
# %%
from sklearn.model_selection import RandomizedSearchCV
parameters = {
    "n_estimators": [50, 100, 200, 300],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "max_depth": [3, 5, 7],
    "subsample": [0.7, 0.8, 1.0],
}

# 2. Add n_iter and random_state
grid = RandomizedSearchCV(
    estimator=GradientBoostingClassifier(random_state=42),
    param_distributions=parameters,
    n_iter=10,  # Number of parameter settings sampled
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    random_state=42,
)
# %%
grid.fit(X_train, Y_train)
y_pred = grid.predict(X_test)
# %%
accuracy_score(Y_test,y_pred)
# %%
grid.best_params_
# %%
