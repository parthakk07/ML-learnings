# %%
import pandas as pd
import numpy as np
# %%
df=pd.read_csv("/home/parthak/Videos/autoYT/project2/train.csv")
df.sample(5)
# %%
df.isnull().mean()*100
# %%
df.describe()
# %%
df.duplicated().sum()
# %%
import seaborn as sns
# %%
sns.pairplot(df)
# %%
sns.histplot(df["gender"])
# %%
sns.histplot(df["academic_work_impact"])
# %%
sns.boxenplot(x=df["social_media_hours"])

# %%
sns.boxenplot(x=df["work_study_hours"])
# %%
df.sample(5)
# %%
sns.histplot(df["addicted_label"])
# %%
df["gender"].unique()

# %%
df.info()
# %%
temp_df=df.drop(columns=["gender","stress_level","academic_work_impact","addicted_label","id"],axis=1)
temp_df.sample(4)
# %%
sns.heatmap(temp_df.corr())
# %%
X=df.drop(["id","addicted_label"],axis=1)
y=df["addicted_label"]
# %%
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.2,random_state=42)
X_train
# %%
# age                         4.184307
# daily_screen_time_hours    13.864376
# social_media_hours         19.381112
# gaming_hours               18.343461
# work_study_hours            7.451592
# sleep_hours                 6.433612
# notifications_per_day       9.775388
# app_opens_per_day          11.673940
# weekend_screen_time        16.208855
#
# gender                      4.199494
# stress_level                7.976638
# academic_work_impact        6.396584
#
# %%
colls=["weekend_screen_time","app_opens_per_day", "notifications_per_day", "sleep_hours","work_study_hours", "gaming_hours", "social_media_hours", "daily_screen_time_hours"]
X_train[colls]=X_train[colls].fillna(0)
X_test[colls]=X_test[colls].fillna(0)
X_train["age"]=X_train["age"].fillna(np.mean(X_train["age"]))
X_test["age"]=X_test["age"].fillna(np.mean(X_train["age"]))
colls_cat=["gender","stress_level","academic_work_impact"]
X_train[colls_cat]=X_train[colls_cat].fillna("missing")
X_test[colls_cat]=X_test[colls_cat].fillna("missing")
X_train
# %%
# %%
from sklearn.preprocessing import OrdinalEncoder , StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
oe=OrdinalEncoder(categories=[["missing","Low","Medium","High"]])
ohe=OneHotEncoder()
std=StandardScaler()
transform=ColumnTransformer([("oe",oe,["stress_level"]),("ohe",ohe,["gender","academic_work_impact"])],remainder="passthrough")
# %%
X_train=transform.fit_transform(X_train)
X_test=transform.transform(X_test)
X_train
# %%
from sklearn.preprocessing import LabelEncoder
label=LabelEncoder()
Y_train=label.fit_transform(Y_train)
Y_test=label.transform(Y_test)
# %%
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
lr=LogisticRegression()
dt=DecisionTreeClassifier()
svc=SVC()
rf=RandomForestClassifier()
lr.fit(X_train,Y_train)
y_pred1=lr.predict(X_test)
dt.fit(X_train,Y_train)
y_pred2=dt.predict(X_test)
rf.fit(X_train,Y_train)
y_pred4=rf.predict(X_test)
# %%
from sklearn.metrics import classification_report, accuracy_score
accuracy_score(Y_test,y_pred1)
# %%
accuracy_score(Y_test,y_pred2)
# %%
accuracy_score(Y_test,y_pred4)
# %%
from sklearn.model_selection import GridSearchCV
parameters={"n_estimators":[10,100, 200,300],"max_depth":[1,5,10,16,None],"min_samples_split":[2,5,10],"min_samples_leaf": [1, 2, 4]}
grid = GridSearchCV(RandomForestClassifier(),param_grid=parameters,cv=5,n_jobs=-1,scoring="accuracy")
grid.fit(X_train,Y_train)
# %%
grid.best_params_
# %%
grid.best_score_
# %%
