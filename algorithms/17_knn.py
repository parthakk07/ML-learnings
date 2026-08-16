# %%
import pandas as pd
import numpy as np
# %%
df=pd.read_csv("/home/parthak/Videos/autoYT/brest_cancer.csv")
df.sample(5)
# %%
df.drop(columns=['id','Unnamed: 32'],inplace=True)
df.head()

# %%
X=df.drop(["diagnosis"],axis=1)
y=df["diagnosis"]
# %%
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=32,random_state=22)
X_train
# %%
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
# %%
from sklearn.neighbors import KNeighborsClassifier
clf=KNeighborsClassifier(n_neighbors=43)
clf.fit(X_train,Y_train)
y_pred=clf.predict(X_test)
# %%
from sklearn.metrics import confusion_matrix
confusion_matrix(Y_test,y_pred)
# %%
from sklearn.metrics import accuracy_score
accuracy_score(Y_test,y_pred)
# %%
import seaborn as sns
sns.scatterplot(x=df["perimeter_mean"],y=df["symmetry_mean"],hue =y)
# %%
scores = []

for i in range(1,16):

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train,Y_train)

    y_pred = knn.predict(X_test)

    scores.append(accuracy_score(Y_test, y_pred))
# %%
import matplotlib.pyplot as plt

plt.plot(range(1,16),scores)

# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
from sklearn.preprocessing import StandardScaler
from ipywidgets import interact, fixed

def load_data():
    cancer = datasets.load_breast_cancer()
    return cancer

def plot_decision_boundaries(n_neighbors, data, labels):
    h = .02
    cmap_light = ListedColormap(['orange', 'blue'])
    cmap_bold = ListedColormap(['darkorange', 'darkblue'])

    clf = neighbors.KNeighborsClassifier(n_neighbors)
    clf.fit(data, labels)

    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(8, 6))
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap=cmap_bold, edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(f'2-Class classification (k = {n_neighbors})')
    plt.show()

cancer = load_data()

# Use only the first two features and standardize them.
X = StandardScaler().fit_transform(cancer.data[:, :2])
y = cancer.target

# Interactive widget
interact(plot_decision_boundaries, n_neighbors=(1, 20), data=fixed(X), labels=fixed(y));
# %%
# support vector machiine
from sklearn.svm import SVC
clf1=SVC()
clf1.fit(X_train,Y_train)
y_pred1=clf1.predict(X_test)
# %%
confusion_matrix(Y_test,y_pred1)
# %%
from sklearn.tree import DecisionTreeClassifier
clf3=DecisionTreeClassifier()
clf3.fit(X_train,Y_train)
y_pred3=clf3.predict(X_test)
# %%
confusion_matrix(Y_test,y_pred3)
