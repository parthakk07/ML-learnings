# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import optuna
# %%
df=pd.read_csv("/home/parthak/Videos/autoYT/project4/train.csv")
df.sample(4)
# %%
df.drop("id",inplace=True,axis=1)
# %%
df.sample(5)
# %%
df.info()
# %%
df.isnull().mean()*100
# %%
df.describe()
# %%
df.duplicated().sum()
# %%
sns.histplot(df["health_condition"])
# %%
sns.histplot(df["diet_type"])
# %%
sns.histplot(df["sleep_quality"])
# %%
sns.histplot(df["stress_level"])
# %%
sns.histplot(df["smoking_alcohol"])
# %%
sns.histplot(df["gender"])
# %%
sns.kdeplot(df["sleep_duration"])
# %%
sns.kdeplot(df["heart_rate"])
# %%

sns.kdeplot(df["bmi"])
# %%
sns.kdeplot(df["calorie_expenditure"])
# %%
sns.kdeplot(df["step_count"])
# %%
sns.kdeplot(df["exercise_duration"])
# %%
sns.kdeplot(df["water_intake"])
# %%
cols = df.columns[df.dtypes == float]
sns.heatmap(df[cols].corr())
# %%
# missing values
# outliers=calorie_expenditure ,exercise_duration
# imbalanced dataset
sns.boxplot(x=df["calorie_expenditure"])
# %%
sns.boxplot(x=df["exercise_duration"])
# %%
# sns.pairplot(df)
# %%
import matplotlib.pyplot as plt
df.hist(bins=50, figsize=(20,15))
plt.show()
# %%
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score ,balanced_accuracy_score ,confusion_matrix ,classification_report
from sklearn.model_selection import train_test_split , cross_val_score
from sklearn.preprocessing import StandardScaler , OrdinalEncoder , OneHotEncoder , LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
# %%
X=df.drop("health_condition",axis=1)
y=df["health_condition"]
X
# %%
X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)
X_train.shape
# %%
outliers_col=["calorie_expenditure","exercise_duration"]
for i in outliers_col:
    min=X_train[i].quantile(0.25)
    max=X_train[i].quantile(0.75)
    iqr=max-min
    lower=min - (1.5*iqr)
    high=max +(1.5 *iqr)
    mask = (X_train[i] > lower) & (X_train[i] < high)
    X_train = X_train[mask]
    Y_train = Y_train[mask]
X_train.shape
# skipping for better accuracy
# %%
num_pipeline = Pipeline([
    ("si", SimpleImputer(strategy="median")),
    ("sc", StandardScaler()),
])
ohe_pipeline = Pipeline([
    ("si", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder()),
])
oe_pipeline = Pipeline([
    ("si", SimpleImputer(strategy="constant", fill_value="missing")),
    ("oe", OrdinalEncoder(categories=[
        ["sedentary", "moderate", "active"],
        ["low", "medium", "high"],
        ["poor", "average", "good"]], handle_unknown="use_encoded_value", unknown_value=-1)),
])
trf = ColumnTransformer([
    ("num", num_pipeline, cols),
    ("ohe", ohe_pipeline, ["gender", "smoking_alcohol", "diet_type"]),
    ("oe", oe_pipeline, ["physical_activity_level", "stress_level", "sleep_quality"]),
], remainder="passthrough")

X_train=trf.fit_transform(X_train)
X_test=trf.transform(X_test)
pd.DataFrame(X_train)
# %%

le=LabelEncoder()
Y_train=le.fit_transform(Y_train)
Y_test=le.transform(Y_test)
# %%
lr=LogisticRegression(class_weight={0:50,1:1},n_jobs=-1)
lr.fit(X_train,Y_train)
y_pred=lr.predict(X_test)
print(balanced_accuracy_score(Y_test,y_pred))
print(classification_report(Y_test,y_pred))
# %%
rf=RandomForestClassifier(n_estimators=104, max_depth=25,min_samples_split=8,min_samples_leaf=1,bootstrap=True,n_jobs=-1,)
rf.fit(X_train,Y_train)
y_pred=rf.predict(X_test)
print(balanced_accuracy_score(Y_test,y_pred))
print(accuracy_score(Y_test,y_pred))
print(classification_report(Y_test,y_pred))
# %%
xg=XGBClassifier()
xg.fit(X_train,Y_train)
y_pred=xg.predict(X_test)
print(balanced_accuracy_score(Y_test,y_pred))
print(accuracy_score(Y_test,y_pred))
print(classification_report(Y_test,y_pred))
# %%
def objective_rf(trial):
    params={"n_estimators":trial.suggest_int("n_estimators",100,300),
        "max_depth":trial.suggest_int("max_depth",5,30),
        "n_jobs":-1,
        "min_samples_split":trial.suggest_int("min_samples_split",2,10),
        "min_samples_leaf":trial.suggest_int("min_samples_leaf",1,5),
        "random_state":42}
    model =RandomForestClassifier(**params)
    score= np.mean(cross_val_score(model,X_train,Y_train,cv=3,n_jobs=-1,scoring="accuracy"))
    return score
# %%
study=optuna.create_study(direction="maximize")
study.optimize(objective_rf,n_trials=10)
print(study.best_params)

# %%
def objective_xg(trial):
    params={"n_estimators":trial.suggest_int("n_estimators",100,400),
            "lambda": trial.suggest_float('lambda', 1e-8, 1.0, log=True),
            "alpha": trial.suggest_float('alpha', 1e-8, 1.0, log=True),
            "gamma": trial.suggest_float('gamma', 1e-8, 1.0, log=True),
            'eta': trial.suggest_float('eta', 0.01, 0.3),
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.4, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0)}

    model=XGBClassifier(**params ,random_state=42,n_jobs=-1)
    score=np.mean(cross_val_score(model,X_train,Y_train,cv=3,n_jobs=-1,scoring="accuracy"))
    return score

study_xg=optuna.create_study(direction="maximize")
study_xg.optimize(objective_xg,n_trials=10)
print(study_xg.best_params)

print("="*25)

xg=XGBClassifier(**study_xg.best_params,random_state=42,n_jobs=-1)
xg.fit(X_train,Y_train)
y_pred=xg.predict(X_test)
print(balanced_accuracy_score(Y_test,y_pred))
print(accuracy_score(Y_test,y_pred))
print(classification_report(Y_test,y_pred))
# %%
# submission
test=pd.read_csv("/home/parthak/Videos/autoYT/project4/test.csv")
X_test=test.drop("id",axis=1)
X_test=trf.transform(X_test)
Y_pred=xg.predict(X_test)
Y_pred=le.inverse_transform(Y_pred)
# %%
submission=pd.DataFrame({"id": test["id"],"health_condition":Y_pred})
submission.to_csv('submission.csv', index=False)
# %%
submission_df = pd.read_csv('submission.csv')
print(submission_df.head(10))
print("\nPrediction Counts:")
print(submission_df['health_condition'].value_counts())

# %%
