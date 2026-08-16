# %%
import pandas as pd

# %%
df = pd.read_csv("/home/parthak/Videos/autoYT/titanic/train.csv")

# %%
from pandas_profiling import ProfileReport

prof = ProfileReport(df)
prof.to_file(output_file="output.html")

# %%
