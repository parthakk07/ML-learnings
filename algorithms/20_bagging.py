# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification
from sklearn.svm import SVC
# %%
X,y = make_classification(n_samples=10000, n_features=10,n_informative=3)
# %%
X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.2,random_state=44)
X_train
# %%
from sklearn.metrics import accuracy_score
dt=DecisionTreeClassifier()
dt.fit(X_train,Y_train)
y_pred=dt.predict(X_test)
accuracy_score(Y_test,y_pred)
# %%
bag=BaggingClassifier(estimator=DecisionTreeClassifier(),n_estimators=100,n_jobs=-1,random_state=42,max_samples=0.5)
bag.fit(X_train,Y_train)
y_pred1=bag.predict(X_test)
accuracy_score(Y_test,y_pred1)
# %%
bag=BaggingClassifier(estimator=SVC(),n_estimators=100,n_jobs=-1,random_state=42,max_samples=0.5)
bag.fit(X_train,Y_train)
y_pred1=bag.predict(X_test)
accuracy_score(Y_test,y_pred1)
# %%
# knn
bag=BaggingClassifier(estimator=KNeighborsClassifier(),n_estimators=100,n_jobs=-1,random_state=42,max_samples=0.5)
bag.fit(X_train,Y_train)
y_pred1=bag.predict(X_test)
accuracy_score(Y_test,y_pred1)

# %%
# pasting

bag = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=500,
    max_samples=0.25,
    bootstrap=False,
    random_state=42,
    verbose = 1,
    n_jobs=-1
)

bag.fit(X_train,Y_train)
y_pred = bag.predict(X_test)
print("Pasting classifier",accuracy_score(Y_test,y_pred))
# %%
# random subspacing
bag = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=500,
    max_samples=1.0,
    bootstrap=False,
    max_features=0.5,
    bootstrap_features=True,
    random_state=42
)
bag.fit(X_train,Y_train)
y_pred = bag.predict(X_test)
print("Pasting classifier",accuracy_score(Y_test,y_pred))
# %%
# random paches
bag = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=500,
    max_samples=0.25,
    bootstrap=True,
    max_features=0.5,
    bootstrap_features=True,
    random_state=42,n_jobs=-1
)
bag.fit(X_train,Y_train)
y_pred = bag.predict(X_test)
print("Pasting classifier",accuracy_score(Y_test,y_pred))
# %%
# oob score
bag = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=500,
    max_samples=0.25,
    bootstrap=True,
    oob_score=True,
    random_state=42,n_jobs=-1
)
bag.fit(X_train,Y_train)
y_pred = bag.predict(X_test)
print("Pasting classifier",accuracy_score(Y_test,y_pred))
# %%
from  sklearn.model_selection import GridSearchCV
parameters = {
    'n_estimators': [50,100,500],
    'max_samples': [0.1,0.4,0.7,1.0],
    'bootstrap' : [True,False],
    'max_features' : [0.1,0.4,0.7,1.0],
    "n_jobs":[-1]
    }
search = GridSearchCV(BaggingClassifier(), parameters, cv=5)
# %%
search.fit(X_train,Y_train)
# %%
search.best_score_
# %%
search.best_params_
# %%
# regression
# %%
import pandas as pd
import numpy as np
df=pd.read_csv("/home/parthak/Videos/autoYT/boston_house_prices.csv")
df.sample(5)
# %%
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
X=df.drop("MEDV",axis=1)
y=df["MEDV"]
X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.2,random_state=42)
X_train
# %%
lr=LinearRegression()
dt=DecisionTreeRegressor()
knn=KNeighborsRegressor()
lr.fit(X_train,Y_train)
y_pred1=lr.predict(X_test)
dt.fit(X_train,Y_train)
y_pred2=dt.predict(X_test)
knn.fit(X_train,Y_train)
y_pred3=knn.predict(X_test)
# %%
r2_score(Y_test,y_pred1)
# %%
r2_score(Y_test,y_pred2)
# %%
r2_score(Y_test,y_pred3)
# %%
from sklearn.ensemble import BaggingRegressor
bag=BaggingRegressor(estimator=LinearRegression(), n_estimators=100,n_jobs=-1)
bag.fit(X_train,Y_train)
y_pred4=bag.predict(X_test)
# %%
r2_score(Y_test,y_pred4)
# %%
from sklearn.model_selection import GridSearchCV
params = {'estimator': [None, LinearRegression(), KNeighborsRegressor()],
          'n_estimators': [20,50,100],
          'max_samples': [0.5,1.0],
          'max_features': [0.5,1.0],
          'bootstrap': [True, False],
          'bootstrap_features': [True, False]}

bagging_regressor_grid = GridSearchCV(BaggingRegressor(random_state=1, n_jobs=-1), param_grid =params, cv=3, n_jobs=-1, verbose=1)
bagging_regressor_grid.fit(X_train, Y_train)

print('Train R^2 Score : %.3f'%bagging_regressor_grid.best_estimator_.score(X_train, Y_train))
print('Test R^2 Score : %.3f'%bagging_regressor_grid.best_estimator_.score(X_test, Y_test))
print('Best R^2 Score Through Grid Search : %.3f'%bagging_regressor_grid.best_score_)
print('Best Parameters : ',bagging_regressor_grid.best_params_)
# %%
