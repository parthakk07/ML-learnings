# %%
from sklearn.datasets import make_classification
import numpy as np
X, y = make_classification(n_samples=100, n_features=2, n_informative=1,n_redundant=0,
                           n_classes=2, n_clusters_per_class=1, random_state=41,hypercube=False,class_sep=10)
# %%
X
# %%
from sklearn.linear_model import LogisticRegression
lor = LogisticRegression()
lor.fit(X,y)
m = -(lor.coef_[0][0]/lor.coef_[0][1])
b = -(lor.intercept_/lor.coef_[0][1])
x_input1 = np.linspace(-3,3,100)
y_input1 = m*x_input + b

# %%
import seaborn as sns
sns.scatterplot(x=X[:,0],y=X[:,1], hue=y)
# %%
import random
# def perceptron(X,y):
#     X=np.insert(X, 0,1,axis=1)
#     weight=np.ones(X.shape[1])
#     lr=0.01
#     epoch=1000
#     for i in range(epoch):
#         j=random.randint(0,X.shape[1])
#         y_hat=0
#         if np.dot(X[j],weight) >0 :
#             y_hat=1
#         weight = weight + lr*(y[j]-y_hat)*X[j]
#     return weight[0],weight[1:]

# %%
def perceptron(X,y):

    X = np.insert(X,0,1,axis=1)
    weights = np.ones(X.shape[1])
    lr = 0.1

    for i in range(1000):
        j = np.random.randint(0,100)
        y_hat = step(np.dot(X[j],weights))
        weights = weights + lr*(y[j]-y_hat)*X[j]

    return weights[0],weights[1:]

def step(z):
    return 1 if z>0 else 0
# %%
intercept_,coef_ = perceptron(X,y)
print(coef_)
print(intercept_)
# %%
m = -(coef_[0]/coef_[1])
b = -(intercept_/coef_[1])
# %%
import matplotlib.pyplot as plt
x_input = np.linspace(-3,3,100)
y_input = m*x_input + b
plt.figure(figsize=(10,6))
plt.plot(x_input,y_input,color='red',linewidth=3)
plt.scatter(X[:,0],X[:,1],c=y,cmap='winter',s=100)
plt.ylim(-3,2)
# %%
def perceptron(X,y):

    X = np.insert(X,0,1,axis=1)
    weights = np.ones(X.shape[1])
    lr = 0.1

    for i in range(1000):
        j = np.random.randint(0,100)
        y_hat = sigmoid(np.dot(X[j],weights))
        weights = weights + lr*(y[j]-y_hat)*X[j]

    return weights[0],weights[1:]

def sigmoid(z):
    return 1/(1 + np.exp(-z))

# %%
intercept_,coef_ = perceptron(X,y)
m = -(coef_[0]/coef_[1])
b = -(intercept_/coef_[1])
x_input2 = np.linspace(-3,3,100)
y_input2 = m*x_input + b
# %%
plt.figure(figsize=(10,6))
plt.plot(x_input,y_input,color='red',linewidth=3)
plt.plot(x_input1,y_input1,color='black',linewidth=3)
plt.plot(x_input2,y_input2,color='brown',linewidth=3)
plt.scatter(X[:,0],X[:,1],c=y,cmap='winter',s=100)
plt.ylim(-3,2)
