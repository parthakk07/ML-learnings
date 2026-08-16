# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder , OrdinalEncoder ,StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score , f1_score ,confusion_matrix ,recall_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
# %%
df=pd.read_csv("/home/parthak/Videos/autoYT/Employee12.csv")
df.sample(5)
# %%
df.isnull().mean()*100
# %%
df.info()
# %%
df.describe()
# %%
df["exp"]=2026-df["JoiningYear"]
df.drop("JoiningYear",axis=1,inplace=True)
df.sample(4)
# %%
numeric_cols = df.select_dtypes(include='number').columns.tolist()
sns.heatmap(df[numeric_cols].corr())
# %%
sns.histplot(df["exp"])
# %%
sns.histplot(df["City"])
# %%
sns.histplot(df["LeaveOrNot"])
# %%
sns.kdeplot(df["exp"])
# %%
sns.kdeplot(df["Age"])
# %%
sns.histplot(df["Gender"])
# %%

sns.histplot(df["EverBenched"])

# %%
sns.histplot(df["Education"])
# %%
sns.kdeplot(df["PaymentTier"])
# %%
sns.histplot(df["ExperienceInCurrentDomain"])
# %%
X=df.drop("LeaveOrNot",axis=1)
y=df["LeaveOrNot"]
X

# %%
X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.2,random_state=42)
X_train
# %%
X_train["Education"].unique()
# %%
trf=ColumnTransformer([("ohe",OneHotEncoder(),["City","Gender","EverBenched"]),("oe",OrdinalEncoder(categories=[["Bachelors","Masters","PHD"]]),["Education"]),("sc",StandardScaler(),["exp","Age"])],remainder="passthrough")
# %%
X_train=trf.fit_transform(X_train)
X_test=trf.transform(X_test)
pd.DataFrame(X_train)
# %%
lr=LogisticRegression()
dt=DecisionTreeClassifier()
rf=RandomForestClassifier()
svc=SVC()
knn=KNeighborsClassifier()
# %%
models=[lr,dt,rf,svc,knn]
for i in range(len(models)):
    models[i].fit(X_train,Y_train)
    y_pred=models[i].predict(X_test)
    print(models[i])
    print("accuracy ==",accuracy_score(Y_test,y_pred))
    print("f1==",f1_score(Y_test,y_pred))
    print("recall==",recall_score(Y_test,y_pred))
    print("conusion matrix ==",confusion_matrix(Y_test,y_pred))
    print("="*15)
# %%
#t2 error should less svc is good here
parameters = {
    "n_estimators": [50, 100, 200, 300],
    "max_depth": [3, 5, 7,None],
}
grid=GridSearchCV(rf,param_grid=parameters,cv=3,n_jobs=-1,scoring="recall")
# %%
grid.fit(X_train,Y_train)
y_pred = grid.predict(X_test)
accuracy_score(Y_test,y_pred)
# %%
grid.best_params_
# %%
grid.best_score_
# %%
confusion_matrix(Y_test,y_pred)
# %%
from xgboost import XGBClassifier
xgbc=XGBClassifier(n_estimators=50,learning_rate=0.1)
# %%
xgbc.fit(X_train,Y_train)
pred=xgbc.predict(X_test)
# %%
print("accuracy ==",accuracy_score(Y_test,pred))
print("f1==",f1_score(Y_test,pred))
print("recall==",recall_score(Y_test,pred))
print("conusion matrix ==",confusion_matrix(Y_test,pred))
# %%
