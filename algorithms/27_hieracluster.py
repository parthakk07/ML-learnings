# %%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# %%
customer_data = pd.read_csv('/home/parthak/Videos/autoYT/hierarchical-clustering-with-python-and-scikit-learn-shopping-data.csv')
customer_data.shape
# %%
customer_data.head()
# %%
data = customer_data.iloc[:, 3:5].values
# %%
import scipy.cluster.hierarchy as shc

plt.figure(figsize=(10, 7))
plt.title("Customer Dendograms")
dend = shc.dendrogram(shc.linkage(data, method='ward'))
# %%
from sklearn.cluster import AgglomerativeClustering

cluster = AgglomerativeClustering(n_clusters=5, linkage='ward')
labels_=cluster.fit_predict(data)
# %%
labels_
# %%
plt.figure(figsize=(10, 7))
plt.scatter(data[:,0], data[:,1], c=cluster.labels_, cmap='rainbow')
