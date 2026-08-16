# %%
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.impute import SimpleImputer

# %%
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

# %%
df = pd.read_csv("/home/parthak/Videos/autoYT/titanic/train.csv")
df.sample(5)
# %%
# %%
# drop useless columns
df.drop(columns=[" PassengerId", "Name", "Ticket", "Cabin"], inplace=True)
df.head()
# %%
X_train, X_test, Y_train, Y_test = train_test_split(
    df.drop(columns=["Survived"]), df[["Survived"]], test_size=0.2, random_state=42
)
X_train
# %%
# imputer for missing values for age and embarked
trf1 = ColumnTransformer(
    [
        ("impute_age", SimpleImputer(), [2]),
        ("impute_embarked", SimpleImputer(strategy="most_frequent"), [6]),
    ],
    remainder="passthrough",
)

# %%
# encoding in embarked and sex both one hot encoding
trf2 = ColumnTransformer(
    [
        (
            "ohe_sex_embarked",
            OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
            [1, 6],
        )
    ],
    remainder="passthrough",
)

# %%
# scalling of aall the colummns
trf3 = ColumnTransformer([("scale", MinMaxScaler(), slice(0, 10))])

# %%
# select 8 best columns which are then used for traning
trf4 = SelectKBest(score_func=chi2, k=8)
# %%
# model traning
trf5 = DecisionTreeClassifier()
# %%
# create pipe line

# %%
pipe = Pipeline(
    [("trf1", trf1), ("trf2", trf2), ("trf3", trf3), ("trf4", trf4), ("trf5", trf5)]
)

# %%
# Display Pipeline

from sklearn import set_config

set_config(display="diagram")
# %%
# train
pipe.fit(X_train, Y_train)
# %%
# explain code
pipe.named_steps
# %%
# predict
Y_pred = pipe.predict(X_test)
# %%
Y_pred
# %%
from sklearn.metrics import accuracy_score

accuracy_score(Y_test, Y_pred)
# %%
# Cross Validation using Pipeline

# cross validation using cross_val_score
from sklearn.model_selection import cross_val_score

cross_val_score(pipe, X_train, Y_train, cv=5, scoring="accuracy").mean()
# %%
# GridSearch using Pipeline
# gridsearchcv
params = {"trf5__max_depth": [1, 2, 3, 4, 5, None]}
# %%
from sklearn.model_selection import GridSearchCV

grid = GridSearchCV(pipe, params, cv=5, scoring="accuracy")
grid.fit(X_train, y_train)
# %%
grid.best_score_
# %%
grid.best_params_
# %%
# Exporting the Pipeline
#
# %%
# export
import pickle

pickle.dump(pipe, open("pipe.pkl", "wb"))


# production ka code
#
# %%
import pickle

import numpy as np

# %%
pipe = pickle.load(open("pipe.pkl", "rb"))
# %%
# Assume user input
test_input2 = np.array([2, "male", 31.0, 0, 0, 10.5, "S"], dtype=object).reshape(1, 7)

# %%
pipe.predict(test_input2)
