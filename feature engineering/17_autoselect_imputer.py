# %%
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# %%
df = pd.read_csv(
    "/home/parthak/Videos/autoYT/titanic/train.csv",
    usecols=["Age", "Fare", "Survived", "Embarked", "SibSp", "Parch", "Pclass", "Sex"],
)
df.sample(5)
# %%
X = df.drop(columns=["Survived"])
y = df["Survived"]
# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
X_train.sample(4)
# %%
numerical_features = ["Age", "Fare"]
numerical_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)

categorical_features = ["Embarked", "Sex"]
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ]
)
# %%
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)
# %%
clf = Pipeline(
    steps=[("preprocessor", preprocessor), ("classifier", LogisticRegression())]
)
# %%
from sklearn import set_config

set_config(display="diagram")
clf
# %%
param_grid = {
    "preprocessor__num__imputer__strategy": ["mean", "median"],
    "preprocessor__cat__imputer__strategy": ["most_frequent", "constant"],
    "classifier__C": [0.1, 1.0, 10, 100],
}

grid_search = GridSearchCV(clf, param_grid, cv=10)
# %%
grid_search.fit(X_train, y_train)

print(f"Best params:")
print(grid_search.best_params_)
# %%
print(f"Internal CV score: {grid_search.best_score_:.3f}")
# %%
cv_results = pd.DataFrame(grid_search.cv_results_)
cv_results = cv_results.sort_values("mean_test_score", ascending=False)
cv_results[
    [
        "param_classifier__C",
        "param_preprocessor__cat__imputer__strategy",
        "param_preprocessor__num__imputer__strategy",
        "mean_test_score",
    ]
]
# %%
