# %%
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# %%
df=sns.load_dataset("iris")
df.sample(4)

# %%
from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
df["species"]=encoder.fit_transform(df["species"])
# %%
df = df[['sepal_length','petal_length','species']]
# %%
X=df.drop(["species"],axis=1)
y=df["species"]
X
# %%
X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.2,random_state=44)
X_train
# %%

clf = LogisticRegression(solver='lbfgs')
clf.fit(X_train, Y_train)
y_pred = clf.predict(X_test)


# %%
from sklearn.metrics import accuracy_score, confusion_matrix
accuracy_score(Y_test,y_pred)
# %%
confusion_matrix(Y_test,y_pred)
# %%
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions

plot_decision_regions(X.values, y.values, clf, legend=2)

# Adding axes annotations
plt.xlabel('sepal length [cm]')
plt.xlabel('petal length [cm]')
plt.title('Softmax on Iris')

plt.show()
