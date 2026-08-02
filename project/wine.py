# %%
import pandas as pd
import numpy as np
# %%
df=pd.read_csv("/home/parthak/Videos/autoYT/project/winequality-red.csv")
df.sample(5)
# %%
df.isnull().sum()
# %%
import seaborn as sns
sns.boxenplot(x=df["total sulfur dioxide"])
# %%
sns.kdeplot(df["citric acid"])
# %%
sns.kdeplot(df["volatile acidity"])
# %%
sns.kdeplot(df["total sulfur dioxide"])
# %%
df.corr()
# %%
sns.kdeplot(df["alcohol"])
# %%
sns.pairplot(df)
# %%
# from pandas_profiling import ProfileReport
# prof=ProfileReport(df)
# prof.to_file("wine.html")
# %%
sns.heatmap(df.corr())

# %%
X=df.drop("quality",axis=1)
y=df["quality"]
# %%
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.2,random_state=44)
X_train

# %%
y.unique()
# %%
cols=X_train.columns
for i in cols:
    Q1=X_train[i].quantile(0.25)
    Q3=X_train[i].quantile(0.75)
    iqr=Q3-Q1
    high = Q3 + 1.5*(iqr)
    low = Q1 - (1.5*iqr)
    X_train[i] = np.where(X_train[i] > high, high, np.where(X_train[i] < low, low, X_train[i]))
    X_test[i]=np.where(X_test[i] >high , high , np.where(X_test[i] <low , low , X_test[i]))
# %%
std=StandardScaler()
X_train_sca=std.fit_transform(X_train)
X_test_sca=std.transform(X_test)
# %%
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
lr=LogisticRegression()
dt=DecisionTreeClassifier()
# %%
lr.fit(X_train_sca,Y_train)
dt.fit(X_train_sca,Y_train)
y_pred1=lr.predict(X_test_sca)
y_pred2=dt.predict(X_test_sca)
# %%
from sklearn.metrics import confusion_matrix , accuracy_score ,f1_score
confusion_matrix(Y_test,y_pred1)
# %%
print(accuracy_score(Y_test,y_pred1))
# 57
# %%
print(accuracy_score(Y_test,y_pred2))
# %%
from sklearn.model_selection import cross_val_score
np.mean(cross_val_score(dt,X,y,scoring="accuracy",cv=10,n_jobs=-1))
# %%
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
rf.fit(X_train_sca,Y_train)
y_pred3=rf.predict(X_test_sca)
# %%
accuracy_score(Y_test,y_pred3)
# %%
from sklearn.model_selection import GridSearchCV
parameters={"n_estimators":[10,100, 200,300],"max_depth":[1,5,10,16,None],"min_samples_split":[2,5,10],"min_samples_leaf": [1, 2, 4]}
grid = GridSearchCV(RandomForestClassifier(),param_grid=parameters,cv=5,n_jobs=-1,scoring="accuracy")
grid.fit(X_train_sca,Y_train)
# %%
grid.best_params_
# %%
grid.best_score_
# %%
f1_score(Y_test,y_pred1,average="weighted")
# %%
