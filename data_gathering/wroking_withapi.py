# %%
import numpy as np
import pandas as pd

# %%
import requests

url = "https://google-ad-library.p.rapidapi.com/search"

querystring = {"domain": "apple.com", "format": "ALL"}

headers = {
    "x-rapidapi-key": "493a6cf1f4mshe17d94b18e54a6ap106956jsnadc47b10ff8e",
    "x-rapidapi-host": "google-ad-library.p.rapidapi.com",
    "Content-Type": "application/json",
}

response = requests.get(url, headers=headers, params=querystring)

# print(response.json()["ads"])
df = pd.DataFrame(response.json()["ads"])

# %%
df = df.iloc[2]


# %%
df.to_csv("")


# %%
import pandas as pd
import requests
from urllib3.exceptions import ResponseError

url = "https://google-ad-library.p.rapidapi.com/search"

querystring = {"domain": "apple.com", "format": "ALL", "limit": "40"}

headers = {
    "x-rapidapi-key": "57b65f2c5emsh0e99f17b77a435fp19d3f8jsnf6a2ff8ac97e",
    "x-rapidapi-host": "google-ad-library.p.rapidapi.com",
    "Content-Type": "application/json",
}

df = pd.DataFrame()

for i in range(1, 300001):
    response = requests.get(url, headers=headers, params=querystring)
    json_data = response.json()

    if "ads" not in json_data:
        print(f"Response keys: {json_data.keys()}")
        print(f"Response: {json_data}")
        break

    tempdf = pd.DataFrame(json_data["ads"])
    df = pd.concat([df, tempdf], ignore_index=True)


# %%
df.info()
