# %%
import numpy as np
import pandas as pd

# %%
date = pd.read_csv("/home/parthak/Videos/autoYT/orders.csv")
time = pd.read_csv("/home/parthak/Videos/autoYT/messages.csv")
# %%
date.sample(5)
# %%
time.sample(5)
# %%
date.info()
# %%
time.info()

# %%
# agar date ys time object hai toh hume usko change karna hoga
# convertiveng to datetime dataframe
#

# %%
date["date"] = pd.to_datetime(date["date"])

# %%
date.info()
# %%
# show year
date["date_year"] = date["date"].dt.year
date.sample(5)
# %%
# show month
date["date_month_number"] = date["date"].dt.month
date.sample(5)
# %%
# show  month name
date["date_month_name"] = date["date"].dt.month_name()
date.sample(5)
# %%
# show day
date["date_day"] = date["date"].dt.day
date.sample(5)
# %%
# show day of week
date["date_day_week"] = date["date"].dt.dayofweek
date.sample(5)
# %%
# show day of week with name
date["date_day_week_name"] = date["date"].dt.day_name()
date.sample(5)
# %%
# show is weekend or not ?
date["date_weekend"] = np.where(
    date["date_day_week_name"].isin(["Sunday", "Saturday"]), 1, 0
)
date.sample(5)
# %%
# show week
# %%  show week (ISO week number)
date["date_week"] = date["date"].dt.isocalendar().week
date.sample(5)

# %%
# show quater
# show week
date["date_quater"] = date["date"].dt.quarter
date.sample(5)
# %%
# show semester
# %%  show week (ISO week number)
date["date_semester"] = np.where(date["date"].dt.month < 6, 1, 2)
date.sample(5)
# %%
# Extract Time elapsed between dates
# aaj ka time and data ke time me kitna tiem hua hai ??
import datetime

today = datetime.datetime.today()

today
# %%
today - date["date"]

# %%
(today - date["date"]).dt.days
# %%
# Months passed
today = pd.Timestamp("now")
months_passed = np.round((today - date["date"]) / np.timedelta64(30, "D"), 0)

# %%
date.sample(5)
# %%
# Converting to datetime datatype
time["date"] = pd.to_datetime(time["date"])

# %%
time.info()
# %%
time["hour"] = time["date"].dt.hour
time["min"] = time["date"].dt.minute
time["sec"] = time["date"].dt.second

time.head()
# %%
time["time"] = time["date"].dt.time

time.head()
# %%
# in seconds

(today - time["date"]) / np.timedelta64(1, "s")
# %%

# in minutes

(today - time["date"]) / np.timedelta64(1, "m")
# %%
# in hours

(today - time["date"]) / np.timedelta64(1, "h")
