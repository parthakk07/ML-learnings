# %%
import numpy as np
import pandas as pd
import requests

# %%
api = "523d9aa1640473333280547fc12b3763"
url = "https://v3.football.api-sports.io/countries"
headers = {"x-apisports-key": api}

response = requests.get(url, headers=headers)
print(response.json())


"""2018 -2024
    /players -name
    /tropies -no of trophies
    /leagues -
"""


# api se baadh me kaarnunga mushikil lagraisss
#
