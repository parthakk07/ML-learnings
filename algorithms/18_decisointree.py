# %%
import pandas as pd
import numpy as np
# %%
df=pd.read_csv("/home/parthak/Videos/autoYT/project/winequality-red.csv")
df.sample(5)
# %%
X=df.drop("quality",axis=1)
y=df["quality"]
# %%
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.2,random_state=44)
X_train
# %%
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor(max_depth=4)
dt.fit(X_train,Y_train)
y_pred=dt.predict(X_test)
# %%
from sklearn.metrics import r2_score
r2_score(Y_test,y_pred)
# %%
for importance , name in sorted(zip(dt.feature_importances_,X_train.columns),reverse=True):
    print(name ,importance)

# %%
from sklearn.tree import plot_tree
plot_tree(dt)
# %%
import graphviz.backend as be

from sklearn.datasets import *
from dtreeviz.trees import *
from IPython.display import Image, display_svg, SVG
# %%
from sklearn.tree import DecisionTreeClassifier
clas = DecisionTreeClassifier()
iris = load_iris()

X_train = iris.data
y_train = iris.target
clas.fit(X_train, y_train)
# %%
viz = dtreeviz(clas,
               X_train,
               y_train,
               feature_names=iris.feature_names,
               class_names=["setosa", "versicolor", "virginica"])
viz
# %%
