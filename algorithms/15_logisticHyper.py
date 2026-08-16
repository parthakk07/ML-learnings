# %%
import pandas as pd
import numpy as np
import seaborn as sns

# %%
from sklearn.datasets import load_breast_cancer
data=load_breast_cancer()

X=pd.DataFrame(data.data,columns=data.feature_names)
y=data.target
X.sample(5)
# %%
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.2,random_state=22)
X_train
# %%
from sklearn.linear_model import LogisticRegression
clf= LogisticRegression()
clf.fit(X_train,Y_train)
y_pred=clf.predict(X_test)
# %%
from sklearn.metrics import accuracy_score, f1_score,confusion_matrix
accuracy_score(Y_test,y_pred)

# %%
f1_score(Y_test,y_pred)
# %%
confusion_matrix(Y_test,y_pred)
# %%
clf2=LogisticRegression(penalty=None,solver="lbfgs")
clf2.fit(X_train,Y_train)
y_pred2=clf2.predict(X_test)
# %%
confusion_matrix(Y_test,y_pred2)
# better
#
