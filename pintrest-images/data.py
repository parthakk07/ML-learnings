import time

import numpy
import pandas
import requests
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

total = 1000
with sync_playwright() as p:
    browser = p.chromium.launch_persistent_context(
        user_data_dir="pintrest_profile", headless=False
    )
    page = browser.new_page()
    page.goto("https://in.pinterest.com/")
    input("enter ")
    imageurl = set()
    while len(imageurl) != total:
        page.mouse.wheel(0, 500)
        time.sleep(2)
        images = page.query_selector_all("img")
        for img in images:
            src = img.get_attribute("src")
            if src and "http" in src:
                imageurl.add(src)
            if len(imageurl) == total:z
                break

    for i, url in enumerate(imageurl):
        try:
            r = requests.get(url)
            with open(f"pinterest_images/{i}.jpg", "wb") as f:
                f.write(r.content)
        except Exception as e:
            print(e)
    print("done")
    input("enter")
    browser.close()
