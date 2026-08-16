# %%
import pandas as pd
import numpy as np
# %%
df=pd.read_csv("/home/parthak/Videos/autoYT/project3/Top_rated_movies (1).csv")
df.sample(5)
# %%
df.isnull().sum()
# %%
import seaborn as sns
import matplotlib.pyplot as plt
# %%
sns.distplot(df["vote_count"])
# %%
sns.distplot(df["rating"])
# %%
df["original_language"].unique()
# %%
sns.histplot(x=df["original_language"])
# %%
df["release_date"]=pd.to_datetime(df["release_date"])
col=["release_date","vote_count","popularity","rating"]
sns.heatmap(df[col].corr())
# %%
sns.pairplot(df)
# %%
X=df[col].drop(["release_date","rating"],axis=1)
y=df["rating"]
X
# %%
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.preprocessing import power_transform
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score , mean_squared_error
from sklearn.compose import ColumnTransformer
# %%
X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.2,random_state=42)
X_train
# %%
trf=ColumnTransformer([("sc",StandardScaler(),["vote_count","popularity"]),("pt",PowerTransformer(),["popularity","vote_count"])],remainder="passthrough")
# %%
X_train=trf.fit_transform(X_train)
X_test=trf.transform(X_test)
X_train
# %%
lr=LinearRegression()
dt=DecisionTreeRegressor()
svr=SVR()
rf=RandomForestRegressor()
lr.fit(X_train,Y_train)
dt.fit(X_train,Y_train)
svr.fit(X_train,Y_train)
rf.fit(X_train,Y_train)
y_pred1=lr.predict(X_test)
y_pred2=dt.predict(X_test)
y_pred3=svr.predict(X_test)
y_pred4=rf.predict(X_test)
# %%
print(f"{r2_score(Y_test,y_pred1)} , {mean_squared_error(Y_test,y_pred1)}")
# %%
print(f"{r2_score(Y_test,y_pred2)} , {mean_squared_error(Y_test,y_pred2)}")
# %%
print(f"{r2_score(Y_test,y_pred3)} , {mean_squared_error(Y_test,y_pred3)}")
# %%
print(f"{r2_score(Y_test,y_pred4)} , {mean_squared_error(Y_test,y_pred4)}")
# %%
import matplotlib.pyplot as plt

plt.scatter(Y_test, y_pred2)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()])
plt.show()
