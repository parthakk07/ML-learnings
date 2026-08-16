# %%
import numpy as np
import  pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# %%
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=100, n_features=2, n_informative=1,n_redundant=0,n_classes=2, n_clusters_per_class=1, random_state=41,hypercube=False,class_sep=20)

# %%
sns.scatterplot(x=X[:,0],y=X[:,1],hue=y)
# %%
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.2,random_state=44)
X_train
# %%
from sklearn.linear_model import LogisticRegression
log = LogisticRegression(solver='lbfgs')  # penalty='l2' by default
log.fit(X_train,Y_train)
y_pred=log.predict(X_test)
# %%
from sklearn.metrics import accuracy_score
accuracy_score(Y_test,y_pred)
# %%
print(log.coef_)
print(log.intercept_)
# %%
m1 = -(log.coef_[0][0]/log.coef_[0][1])
b1 = -(log.intercept_/log.coef_[0][1])
# %%
x_input = np.linspace(-3,3,100)
y_input = m1*x_input + b1
# %%
def gd(X,y):

    X = np.insert(X,0,1,axis=1)
    weights = np.ones(X.shape[1])
    lr = 0.5

    for i in range(566000):
        y_hat = sigmoid(np.dot(X,weights))
        weights = weights + lr*(np.dot((y-y_hat),X)/X.shape[0])

    return weights[1:],weights[0]

def sigmoid(z):
    return 1/(1 + np.exp(-z))

# %%
coef_,intercept_ = gd(X,y)
m = -(coef_[0]/coef_[1])
b = -(intercept_/coef_[1])
# %%
x_input1 = np.linspace(-3,3,100)
y_input1 = m*x_input1 + b
# %%
plt.figure(figsize=(10,6))
plt.plot(x_input,y_input,color='red',linewidth=3)
plt.plot(x_input1,y_input1,color='black',linewidth=3)
plt.scatter(X[:,0],X[:,1],c=y,cmap='winter',s=100)
plt.ylim(-3,2)
