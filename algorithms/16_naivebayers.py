# %%
import numpy as np
import pandas as pd
# %%
df=pd.read_csv("/home/parthak/Videos/autoYT/play_tennis.csv")
df.sample(4)
# %%
df.drop(["day"],axis=1,inplace=True)
# %%
df.sample(5)
# %%
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(df.drop(["play"],axis=1),df["play"],test_size=0.2,random_state=33)
X_train
# %%
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder,OrdinalEncoder
# Option 2: Use integer column indices instead of strings
transformer = ColumnTransformer([
    ("ohe", OneHotEncoder(), [0, 2, 3]),  # outlook, humidity, wind
    ("ordinal", OrdinalEncoder(categories=[["Cool", "Mild", "Hot"]]), [1])  # temp
])

X_train=transformer.fit_transform(X_train)
X_test=transformer.transform(X_test)
X_test

# %%
trf= LabelEncoder()
Y_train=trf.fit_transform(Y_train)
Y_test=trf.transform(Y_test)
Y_test
# %%
from sklearn.linear_model import LogisticRegression
clf=LogisticRegression()
clf.fit(X_train,Y_train)
y_pred=clf.predict(X_test)
# %%
from sklearn.metrics import accuracy_score
accuracy_score(Y_test,y_pred)
# %%
# thsi iwas logisticregression now navie bayer
#
from sklearn.naive_bayes import GaussianNB
clf=GaussianNB()
clf.fit(X_train,Y_train)
y_pred=clf.predict(X_test)
# %%
from sklearn.metrics import accuracy_score,f1_score, confusion_matrix
accuracy_score(Y_test,y_pred)
# %%
f1_score(Y_test,y_pred)
# %%
confusion_matrix(Y_test,y_pred)
