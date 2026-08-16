# %%
import numpy as np
import pandas as pd

# %%
df = pd.read_csv("/home/parthak/Videos/autoYT/titanic/train.csv")
df.sample(5)
# %%
# drop useless columns
df.drop(columns=[" PassengerId", "Name", "Ticket", "Cabin"], inplace=True)
df.head()
# %%
# test train split
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(
    df.drop(columns=["Survived"]), df["Survived"], test_size=0.3, random_state=20
)
X_train.head()
# %%
df.isnull().sum()
# %%
# simple imputer for missing values
from sklearn.impute import SimpleImputer

si_age = SimpleImputer()
si_embarked = SimpleImputer(strategy="most_frequent")

X_train_age = si_age.fit_transform(X_train[["Age"]])
X_train_embarked = si_embarked.fit_transform(X_train[["Embarked"]])

X_test_age = si_age.transform(X_test[["Age"]])
X_test_embarked = si_embarked.transform(X_test[["Embarked"]])
# %%
X_train_age
# %%
# one hot encoding in gender and embarked
from sklearn.preprocessing import OneHotEncoder

ohe_sex = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
ohe_embarked = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

X_train_sex = ohe_sex.fit_transform(X_train[["Sex"]])
X_train_embarked = ohe_embarked.fit_transform(X_train_embarked)

X_test_sex = ohe_sex.fit_transform(X_test[["Sex"]])
X_test_embarked = ohe_embarked.fit_transform(X_test_embarked)

# %%
X_train_sex
# %%
X_train_rem = X_train.drop(columns=["Sex", "Embarked", "Age"])
X_test_rem = X_test.drop(columns=["Sex", "Embarked", "Age"])
# %%
X_train_transformed = np.concatenate(
    (X_train_rem, X_train_age, X_train_sex, X_train_embarked), axis=1
)
X_test_transformed = np.concatenate(
    (X_test_rem, X_test_age, X_test_sex, X_test_embarked), axis=1
)
# %%
X_train_transformed.shape
# %%
# model tarning
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()
clf.fit(X_train_transformed, Y_train)


# %%


Y_pred = clf.predict(X_test_transformed)
Y_pred


# %%
from sklearn.metrics import accuracy_score

accuracy_score(Y_test, Y_pred)
# %%

import os

# Make sure the output directory exists (no error if it already does)
os.makedirs("models", exist_ok=True)

import pickle

pickle.dump(ohe_sex, open("models/ohe_sex.pkl", "wb"))
pickle.dump(ohe_embarked, open("models/ohe_embarked.pkl", "wb"))
pickle.dump(clf, open("models/clf.pkl", "wb"))

# %%
# predit ya production code me bhout lafda hai like
# pure code reverse me dubaara likho for prediction
# isliye ye pipeline important hai and
# agar kuch change karna hua model me toh pura production code kharab


# yaha mai copy past e kar raha prooduction wala feel le bs
# %%

import pickle

import numpy as np

# %%

ohe_sex = pickle.load(open("models/ohe_sex.pkl", "rb"))
ohe_embarked = pickle.load(open("models/ohe_embarked.pkl", "rb"))
clf = pickle.load(open("models/clf.pkl", "rb"))
# %%
# Assume user input
# Pclass/gender/age/SibSp/Parch/Fare/Embarked
test_input = np.array([2, "male", 31.0, 0, 0, 10.5, "S"], dtype=object).reshape(1, 7)
# %%
test_input
# %%
test_input_sex = ohe_sex.transform(test_input[:, 1].reshape(1, 1))
# %%
test_input_sex
# %%
test_input_embarked = ohe_embarked.transform(test_input[:, -1].reshape(1, 1))
# %%
test_input_embarked
# %%
test_input_age = test_input[:, 2].reshape(1, 1)
# %%
test_input_transformed = np.concatenate(
    (test_input[:, [0, 3, 4, 5]], test_input_age, test_input_sex, test_input_embarked),
    axis=1,
)
# %%
test_input_transformed.shape
# %%
clf.predict(test_input_transformed)

# %%


# ye pura prooduction ka code hai jo ki deploy me jata hai ya jo website par rahega
