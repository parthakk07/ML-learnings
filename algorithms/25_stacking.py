# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# %%
from sklearn.datasets import fetch_openml

# Fetch the dataset natively from OpenML
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist["data"], mnist["target"]
# %%
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.2,random_state=42)
X_train
# %%
from sklearn.ensemble import StackingClassifier
estimators=[("rf",RandomForestClassifier(n_estimators=100)),("svc",SVC()),("dt",DecisionTreeClassifier())]
clf=StackingClassifier(estimators=estimators,cv=5,final_estimator=LogisticRegression())
# %%
clf.fit(X_train,Y_train)
# %%
y_pred=clf.predict(X_test)
# %%
from sklearn.metrics import accuracy_score, confusion_matrix
accuracy_score(Y_test,y_pred)
# %%
confusion_matrix(Y_test,y_pred)
# %%
Y_train = Y_train.astype(int)
Y_test = Y_test.astype(int)

from xgboost import XGBClassifier
xgbc=XGBClassifier()
xgbc.fit(X_train,Y_train)
# %%
y_pred=xgbc.predict(X_test)
# %%
accuracy_score(Y_test,y_pred)
# %%

confusion_matrix(Y_test,y_pred)

# %%
