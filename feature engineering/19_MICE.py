# %%
import numpy as np
import pandas as pd

df = pd.DataFrame(
    {
        "Age": [22, 25, np.nan, 35, 40, np.nan, 50],
        "Salary": [25000, 30000, 45000, np.nan, 70000, 80000, np.nan],
        "Experience": [1, 2, 5, 8, np.nan, 15, 20],
    }
)

print(df)
# %%
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

imputer = IterativeImputer(max_iter=10, random_state=42, verbose=2)

X_imputed = imputer.fit_transform(df)

df_imputed = pd.DataFrame(X_imputed, columns=df.columns)

print(df_imputed)
# %%
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

current = df.copy()

for i in range(1, 6):
    imputer = IterativeImputer(max_iter=i, random_state=42)

    result = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    print(f"\nIteration {i}")
    print(result)

# %%
