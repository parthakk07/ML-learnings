# %%
import pandas as pd
import numpy as np
# %%
df=pd.read_csv("/home/parthak/Videos/autoYT/Iris.csv")
df.sample(5)
# %%
df.drop("Id",axis=1,inplace=True)
# %%
from sklearn.preprocessing import LabelEncoder
df["Species"]=LabelEncoder().fit_transform(df["Species"])
df.head()
# %%
import seaborn as sns
sns.pairplot(df,hue="Species")
# %%
new_df = df[df['Species'] != 0][['SepalLengthCm','SepalWidthCm','Species']]
new_df.head()
# %%
X = df.iloc[:,0:2]
y = df.iloc[:,-1]
# %%
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
# %%
clf1=LogisticRegression()
clf2=KNeighborsClassifier()
clf3=RandomForestClassifier()
# %%
estimators=[("lr",clf1),("knn",clf2),("rf",clf3)]
# %%
for estimnate in estimators:
    score=cross_val_score(estimnate[1],X,y,cv=10,scoring="accuracy")
    print(estimnate[0],np.mean(score))
# %%
# hard vothing
from sklearn.ensemble import VotingClassifier
vc=VotingClassifier(estimators=estimators,voting="hard")
# %%
np.mean(cross_val_score(vc,X,y,cv=10,scoring="accuracy"))
# %%
# soft voting
vc=VotingClassifier(estimators=estimators,voting="soft")
np.mean(cross_val_score(vc,X,y,cv=10,scoring="accuracy"))
# %%
from sklearn.svm import SVC
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=2)

svm1 = SVC(probability=True, kernel='poly', degree=1)
svm2 = SVC(probability=True, kernel='poly', degree=2)
svm3 = SVC(probability=True, kernel='poly', degree=3)
svm4 = SVC(probability=True, kernel='poly', degree=4)
svm5 = SVC(probability=True, kernel='poly', degree=5)

estimators = [('svm1',svm1),('svm2',svm2),('svm3',svm3),('svm4',svm4),('svm5',svm5)]

for estimator in estimators:
    x = cross_val_score(estimator[1],X,y,cv=10,scoring='accuracy')
    print(estimator[0],np.round(np.mean(x),2))
# %%
vc=VotingClassifier(estimators=estimators,voting="soft")
np.mean(cross_val_score(vc,X,y,cv=10,scoring="accuracy"))
# %%
# regression
df=pd.read_csv("/home/parthak/Videos/autoYT/placement2.csv")
df.sample(5)
# %%
X=df.drop("package",axis=1)
y=df["package"]
# %%
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
clf1=LinearRegression()
clf2=SVR()
clf3=DecisionTreeRegressor()
estimators=[("lr",clf1),("svr",clf2),("dt",clf3)]
# %%
for i in estimators:
    print(i[0],np.mean(cross_val_score(i[1],X,y,cv=10,scoring="r2")))
# %%
from sklearn.ensemble import VotingRegressor
# %%
vr=VotingRegressor(estimators=estimators)
scores = cross_val_score(vr,X,y,scoring='r2',cv=10)
print("Voting Regressor",np.round(np.mean(scores),2))
# %%

dt1 = DecisionTreeRegressor(max_depth=1)
dt2 = DecisionTreeRegressor(max_depth=3)
dt3 = DecisionTreeRegressor(max_depth=5)
dt4 = DecisionTreeRegressor(max_depth=7)
dt5 = DecisionTreeRegressor(max_depth=None)

estimators = [('dt1',dt1),('dt2',dt2),('dt3',dt3),('dt4',dt4),('dt5',dt5)]

for estimator in estimators:
  scores = cross_val_score(estimator[1],X,y,scoring='r2',cv=10)
  print(estimator[0],np.round(np.mean(scores),2))
# %%

vr = VotingRegressor(estimators)
scores = cross_val_score(vr,X,y,scoring='r2',cv=10)
print("Voting Regressor",np.round(np.mean(scores),2))
# %%
