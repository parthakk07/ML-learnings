# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_predict, cross_val_score, train_test_split
from sklearn.preprocessing import PowerTransformer

# %%
df = pd.read_csv("/home/parthak/Videos/autoYT/concrete_data.csv")
df.sample(5)
# %%
df.isnull().sum()
# %%
df.describe()
# %%
# 0 hai kuch columns me toh box cox kuch kuch par laga sakte hai
X = df.drop(columns=["Strength"])
Y = df.iloc[:, -1]
X
# %%
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.3, random_state=33
)
# %%
# without transform
clf = LinearRegression()
clf.fit(X_train, Y_train)

y_pred = clf.predict(X_test)

r2_score(Y_test, y_pred)
# %%
clf = LinearRegression()
np.mean(cross_val_score(clf, X, Y, scoring="r2"))
# %%
# halat kharab hai bina transform ke
# %%
X_train.isnull().sum()
# %%
stats.plot()
sns.kdeplot(X_train["Cement"])
# %%
# Plotting the distplots without any transformation

for col in X_train.columns:
    plt.figure(figsize=(14, 4))
    plt.subplot(121)
    sns.kdeplot(X_train[col])
    plt.title(col)

    plt.subplot(122)
    stats.probplot(X_train[col], dist="norm", plot=plt)
    plt.title(col)

    plt.show()

# %%
# applying box cox
pt = PowerTransformer(method="box-cox")
X_train_transformed = pt.fit_transform(X_train + 0.000000001)
X_test_transformed = pt.transform(X_test + 0.00000000001)

# %%
pd.DataFrame({"cols": X_train.columns, "box_cox_lambdas": pt.lambdas_})
# %%
# linera regression model on box-cox
clf = LinearRegression()
clf.fit(X_train_transformed, Y_train)
y_pred2 = clf.predict(X_test_transformed)

r2_score(Y_test, y_pred2)

# %%
# cross val score
pt = PowerTransformer(method="box-cox")
X_transformed = pt.fit_transform(X + 0.0000001)
clf = LinearRegression()

np.mean(cross_val_score(clf, X_transformed, Y, scoring="r2"))
# %%
# 64 persent
# now yea jonson
pt1 = PowerTransformer()

X_train_transformed2 = pt1.fit_transform(X_train)
X_test_transformed2 = pt1.transform(X_test)

lr = LinearRegression()
lr.fit(X_train_transformed2, Y_train)

y_pred3 = lr.predict(X_test_transformed2)

print(r2_score(Y_test, y_pred3))

pd.DataFrame({"cols": X_train.columns, "Yeo_Johnson_lambdas": pt1.lambdas_})
# %%

# cross val score kare final
pt = PowerTransformer()
X_transformed2 = pt.fit_transform(X)

lr = LinearRegression()
np.mean(cross_val_score(lr, X_transformed2, Y, scoring="r2"))
# %%
# 68 hai in yea johnson me so best yei hai use karne ke liye har jagha
# plus ye standardization bhi khud kar deta hai (power transformers dono)
