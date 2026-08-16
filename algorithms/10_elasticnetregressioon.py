# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression,Lasso,ElasticNet ,Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
# %%
data=load_diabetes()
X=data.data
y=data.target
# %%
X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.2,random_state=33)
X_train
# %%
lr=LinearRegression()
lr.fit(X_train,Y_train)
y_pred=lr.predict(X_test)
r2_score(Y_test,y_pred)
# %%
r=Ridge(alpha=0.1)
r.fit(X_train,Y_train)
r2_score(Y_test,r.predict(X_test))
# %%
l=Lasso(alpha=0.1)
l.fit(X_train,Y_train)
r2_score(Y_test,l.predict(X_test))
# %%
enr=ElasticNet(alpha=0.01, l1_ratio=0.9)
enr.fit(X_train,Y_train)
y_pred2=enr.predict(X_test)
r2_score(Y_test,y_pred2)
# %%
df=pd.read_csv("/home/parthak/Videos/autoYT/concrete_data.csv")
df.sample(4)
# %%
X= df.drop(["Strength"],axis=1)
y=df["Strength"]
X
# %%
X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.2,random_state=44)
X_train
# %%
lr=LinearRegression()
lr.fit(X_train,Y_train)
r2_score(Y_test,lr.predict(X_test))
# %%
r=Ridge(alpha=100000)
r.fit(X_train,Y_train)
r2_score(Y_test,r.predict(X_test))
# %%
l=Lasso(alpha=100)
l.fit(X_train,Y_train)
r2_score(Y_test,l.predict(X_test))
# %%
enr=ElasticNet(alpha=100000, l1_ratio=0.9)
enr.fit(X_train,Y_train)
y_pred2=enr.predict(X_test)
r2_score(Y_test,y_pred2)
