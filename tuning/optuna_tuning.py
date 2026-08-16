# %%
import pandas as pd
import numpy as np
import optuna
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
# %%
# Load the Pima Indian Diabetes dataset (from UCI repository)
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
           'DiabetesPedigreeFunction', 'Age', 'Outcome']

# Load the dataset
df = pd.read_csv(url, names=columns)

df.head()

# %%
df.isnull().sum()

# %%
cols=["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]
df[cols]=df[cols].replace(0,np.nan)
df.fillna(df.mean(),inplace=True)
df.isnull().sum()
# %%
df.head()
# %%
X=df.drop("Outcome",axis=1)
y=df["Outcome"]
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_teest=train_test_split(X,y,test_size=0.2,random_state=42)
X_train
# %%
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
# %%
print(f'Training set shape: {X_train.shape}')
print(f'Test set shape: {X_test.shape}')
# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 50, 200)
    max_depth = trial.suggest_int('max_depth', 3, 20)
    model=RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth,random_state=42)
    score=cross_val_score(model,X_train,Y_train,cv=5,scoring="accuracy").mean()
    return score
# %%
# Create a study object and optimize the objective function
study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())  # We aim to maximize accuracy
study.optimize(objective, n_trials=50)  # Run 50 trials to find the best hyperparameters
# %%
print(f'Best trial accuracy: {study.best_trial.value}')
print(f'Best hyperparameters: {study.best_trial.params}')
# %%
rf=RandomForestClassifier(n_estimators=188,max_depth=7,random_state=42)
rf.fit(X_train,Y_train)
y_pred=rf.predict(X_test)
# %%
from sklearn.metrics import accuracy_score
accuracy_score(Y_teest,y_pred)
# %%
# For visualizations
from optuna.visualization import plot_optimization_history, plot_parallel_coordinate, plot_slice, plot_contour, plot_param_importances
# %%
# 1. Optimization History
plot_optimization_history(study).show()
# %%
# 2. Parallel Coordinates Plot
plot_parallel_coordinate(study).show()
# %%
