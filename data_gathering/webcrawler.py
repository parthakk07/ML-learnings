# %%
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

# %%
# print(requests.get("https://www.naukri.com/machine-learning-engineer-jobs?k=machine%20learning%20engineer").text) it still works withot header

# %%
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 6.3; Win 64 ; x64) Apple WeKit /537.36(KHTML , like Gecko) Chrome/80.0.3987.162 Safari/537.36"
}
webpage = requests.get(
    "https://www.gutenberg.org/ebooks/search/?sort_order=downloads", headers=headers
)

# %%
soup = BeautifulSoup(webpage.text, "lxml")
print(soup)
# %%
print(soup.prettify())

# %%
books = soup.find_all("a", class_="link")
# %%
book_name = []
author = []
downloads = []

for i in books:
    book_name.append(i.find_all("span", class_="title")[0].text.strip())
    author.append(i.find_all("span", class_="subtitle")[0].text.strip())
    downloads.append(i.find_all("span", class_="extra")[0].text.strip())

df = pd.DataFrame({"books": book_name, "author": author, "downloads": downloads})

# %%
# print(book_name)
# print(author)
# print(downloads)
df.info()

# %%
final_df = pd.DataFrame({})
for j in range(0, 625, 25):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 6.3; Win 64 ; x64) Apple WeKit /537.36(KHTML , like Gecko) Chrome/80.0.3987.162 Safari/537.36"
    }
    webpage = requests.get(
        f"https://www.gutenberg.org/ebooks/search/?sort_order=downloads&start_index={j}",
        headers=headers,
    )
    soup = BeautifulSoup(webpage.text, "lxml")
    books = soup.find_all("a", class_="link")
    book_name = []
    author = []
    downloads = []

    for i in books:
        try:
            book_name.append(i.find("span", class_="title").text.strip())
        except:
            book_name.append(np.nan)
        try:
            author.append(i.find("span", class_="subtitle").text.strip())
        except:
            author.append(np.nan)
        try:
            downloads.append(i.find("span", class_="extra").text.strip())
        except:
            downloads.append(np.nan)

    df = pd.DataFrame({"books": book_name, "author": author, "downloads": downloads})
    final_df = pd.concat([final_df, df], ignore_index=True)
    print(final_df)


# %%
final_df.info()
final_df.to_csv("books.csv")
