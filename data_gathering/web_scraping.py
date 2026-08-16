# %%
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

# %%
final_df = pd.DataFrame({})
for i in range(1, 500):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 6.3; Win 64 ; x64) Apple WeKit /537.36(KHTML , like Gecko) Chrome/80.0.3987.162 Safari/537.36"
    }
    webpage = requests.get(
        f"https://www.ambitionbox.com/list-of-companies?page={i}", headers=headers
    )
    soup = BeautifulSoup(webpage.text, "lxml")
    company = soup("div", class_="companyCardWrapper")
    for j in company:
        company_name = []
        companyCard = []
        highratedfor = []
        criticalratedfor = []
        rating = []
        reviews = []
        salaries = []
        interviews = []
        jobs = []
        try:
            company_name.append(j.find("h2").text.strip())
        except:
            company_name.append(np.nan)
        try:
            companyCard.append(
                j.find("span", class_="companyCardWrapper__interLinking").text.strip()
            )
        except:
            companyCard.append(np.nan)
        try:
            highratedfor.append(
                j.find_all("span", class_="companyCardWrapper__ratingValues")[
                    0
                ].text.strip()
            )
        except:
            highratedfor.append(np.nan)
        try:
            criticalratedfor.append(
                j.find_all("span", class_="companyCardWrapper__ratingValues")[
                    1
                ].text.strip()
            )
        except:
            criticalratedfor.append(np.nan)
        try:
            rating.append(
                j.find("div", class_="rating_text rating_text--md").text.strip()
            )
        except:
            rating.append(np.nan)
        try:
            reviews.append(
                j.find_all("span", class_="companyCardWrapper__ActionCount")[
                    0
                ].text.strip()
            )
        except:
            reviews.append(np.nan)
        try:
            salaries.append(
                j.find_all("span", class_="companyCardWrapper__ActionCount")[
                    1
                ].text.strip()
            )
        except:
            salaries.append(np.nan)
        try:
            interviews.append(
                j.find_all("span", class_="companyCardWrapper__ActionCount")[
                    2
                ].text.strip()
            )
        except:
            interviews.append(np.nan)
        try:
            jobs.append(
                j.find_all("span", class_="companyCardWrapper__ActionCount")[
                    3
                ].text.strip()
            )
        except:
            jobs.append(np.nan)

        # print(company_name)
        # print(companyCard)
        # print(highratedfor)
        # print(criticalratedfor)
        # print(rating)
        # print(reviews)
        # print(salaries)
        # print(interviews)
        # print(jobs)

        df = pd.DataFrame(
            {
                "Company": company_name,
                "highratedfor": highratedfor,
                "criticalratedfor": criticalratedfor,
                "rating": rating,
                "reviews": reviews,
                "salaries": salaries,
                "interviews": interviews,
                "jobs": jobs,
                "companyCard": companyCard,
            }
        )
        df["branches"] = "NA"
        final_df = pd.concat([final_df, df], ignore_index=True)
        print(final_df)

# %%
final_df.info()
final_df.to_csv("crawl.csv")
final_df
