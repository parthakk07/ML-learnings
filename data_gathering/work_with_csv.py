# %%
import numpy as np
import pandas as pd

# %%
df = pd.read_csv("/home/parthak/Videos/autoYT/titanic/train.csv")
df

# %%
# opening a csv file from url
from io import StringIO

import requests

url = ""
headers = {"User-Agent": "Mozilla/5.0"}
req = requests.get(url, headers=headers)
data = StringIO(req.text)
pd.read_csv(data)
# %%
# tsv files tab seperated value
# sep is used for seperation of the data
# names to give a columns a name
df = pd.read_csv(
    "file.tsv",
    sep="\t",
    names=["bot", "number", "ek aur number", "rollno"],
    index_col="bot",
)
df
# %%
pd.read_csv("file.tsv", sep="\t", nrows=100)  # restrict the number of rows
# %%
pd.read_csv("file.tsv", sep="\t", skiprows=[1, 3, 4])  # skips the rows
# %%
pd.read_csv(
    "file.tsv", sep="\t", header=1, usecols=[0]
)  # header change the columns name for default header =0first row in column name used in something other case
# use columns used for extracting only columns needed
# encoding parameters - data sety have default encoding (UTF- 8)
# sometimes data set having different encoding ex emoji
# change encoding in sublime or encoding param me usko format dal toh

# %%
pd.read_csv("zomato.csv", encoding="latin-1")
# %%
# skip bad lines
df = pd.read_csv("file.csv", on_bad_lines="skip")
df
# %%
# dtype amd astype to change data type of a specifc columns
df = pd.read_csv(
    "/home/parthak/Videos/autoYT/titanic/train.csv", dtype={"Survived": float}
)
df.info()
# %%
# in default date comes  in obbjectss/str to convert in to date
df = pd.read_csv(
    "/home/parthak/Videos/autoYT/practice/patients.csv", parse_dates=["birthdate"]
).info()


# converters
# %%
def rename(gender):
    if gender == "male":
        return "Femael"
    return "female"


pd.read_csv(
    "/home/parthak/Videos/autoYT/practice/patients.csv",
    converters={"assigned_sex": rename},
)
# %%
# na_values = make value nan in a column
pd.read_csv("/home/parthak/Videos/autoYT/practice/patients.csv", na_values=["male"])
# %%
dfs = pd.read_csv("/home/parthak/Videos/autoYT/practice/patients.csv", chunksize=500)
for chunks in dfs:
    print(chunks)
