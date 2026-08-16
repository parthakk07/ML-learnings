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
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score
abc=AdaBoostClassifier()
abc.fit(X,y)
np.mean(cross_val_score(abc,X,y,cv=10,scoring="accuracy"))
# %%
from sklearn.model_selection import GridSearchCV
parameters={"n_estimators":[10,50,100,500],"learning_rate":[0.0001,0.001,0.01,0.1,1.0]}
best=GridSearchCV(AdaBoostClassifier(),param_grid=parameters,cv=10,scoring="accuracy",n_jobs=-1)
# %%

best.fit(X,y)
# %%
best.best_score_
# %%
best.best_params_
# %%
