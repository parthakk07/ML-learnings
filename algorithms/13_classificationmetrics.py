# %%
import pandas as pd
import numpy as np
# %%
df=pd.read_csv("/home/parthak/Videos/autoYT/heart.csv")
df.sample(5)

# %%
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(df.iloc[:,0:-1],df.iloc[:,-1],test_size=0.2,random_state=2)
# %%
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
clf1=LogisticRegression()
clf2=DecisionTreeClassifier()
clf1.fit(X_train,Y_train)
clf2.fit(X_train,Y_train)
y_pred1=clf1.predict(X_test)
y_pred2=clf2.predict(X_test)
# %%
from sklearn.metrics import accuracy_score
accuracy_score(Y_test,y_pred1)
# %%
accuracy_score(Y_test,y_pred2)
# %%
from sklearn.metrics import confusion_matrix
confusion_matrix(Y_test, y_pred1)
# %%
confusion_matrix(Y_test, y_pred2)
# %%
from sklearn.metrics import precision_score,recall_score,f1_score
precision_score(Y_test,y_pred1)
# %%
precision_score(Y_test,y_pred2)
# %%
recall_score(Y_test,y_pred1)
# %%
recall_score(Y_test,y_pred1)
# %%
f1_score(Y_test,y_pred1)
# %%
f1_score(Y_test,y_pred1)
