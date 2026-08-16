# %%
import pandas as pd
import numpy as np
# %%
df=pd.read_csv("/home/parthak/Videos/autoYT/heart.csv")
df.sample(5)
# %%
X=df.drop("target",axis=1)
y=df["target"]
# %%
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.2,random_state=42)
X_train
# %%
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
lr=LogisticRegression()
dt=DecisionTreeClassifier()
knn=KNeighborsClassifier()
svc=SVC()
rf=RandomForestClassifier()
# %%
lr.fit(X_train,Y_train)
y_pred=lr.predict(X_test)
accuracy_score(Y_test,y_pred)
# %%
dt.fit(X_train,Y_train)
y_pred1=dt.predict(X_test)
accuracy_score(Y_test,y_pred1)
# %%
knn.fit(X_train,Y_train)
y_pred2=knn.predict(X_test)
accuracy_score(Y_test,y_pred2)
# %%
rf.fit(X_train,Y_train)
y_pred3=rf.predict(X_test)
accuracy_score(Y_test,y_pred3)
# %%
svc.fit(X_train,Y_train)
y_pred4=svc.predict(X_test)
accuracy_score(Y_test,y_pred4)
# %%
rf=RandomForestClassifier(n_estimators=1000,random_state=42,n_jobs=-1)
rf.fit(X_train,Y_train)
y_pred3=rf.predict(X_test)
accuracy_score(Y_test,y_pred3)
# %%
rf=RandomForestClassifier(oob_score=True)
rf.fit(X_train,Y_train)
y_pred3=rf.predict(X_test)
accuracy_score(Y_test,y_pred3)
# %%
rf.oob_score_
# %%
