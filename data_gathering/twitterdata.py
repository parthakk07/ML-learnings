# %%
"""
Twitter/X Follower Scraper — FINAL VERSION
- Fixes lazy loading stopping
- Human-like scrolling
- Multiple fallback strategies
- Scrapes ENTIRE list

Run: python scraper_final.py
"""

import asyncio
import json
import csv
import random
from datetime import datetime
from pathlib import Path

from playwright.async_api import async_playwright


class TwitterFollowerScraper:
    def __init__(self, cookies_file: str = 'twitter_cookies.json'):
        self.cookies_file = cookies_file
        self.data = []

    async def login_and_save_cookies(self, page):
        print("=" * 60)
        print("BROWSER OPENED — PLEASE LOG IN TO TWITTER/X MANUALLY")
        print("=" * 60)
        print("1. Enter your username/email and password")
        print("2. Complete any 2FA if prompted")
        print("3. Once you see your HOME TIMELINE, press ENTER in this terminal")
        print("=" * 60)

        await page.goto('https://x.com/i/flow/login', wait_until='domcontentloaded')
        input("\n👉 Press ENTER after you've logged in and see your timeline...")

        cookies = await page.context.cookies()
        with open(self.cookies_file, 'w') as f:
            json.dump(cookies, f, indent=2)
        print(f"✅ Cookies saved to {self.cookies_file}")

    async def load_cookies(self, page):
        if Path(self.cookies_file).exists():
            with open(self.cookies_file, 'r') as f:
                cookies = json.load(f)
            await page.context.add_cookies(cookies)
            print(f"✅ Loaded cookies from {self.cookies_file}")
            return True
        return False

    async def human_scroll(self, page):
        """Simulate human-like scrolling to trigger lazy loading."""
        # Small random scrolls
        for _ in range(random.randint(2, 4)):
            amount = random.randint(300, 700)
            await page.evaluate(f'window.scrollBy(0, {amount})')
            await page.wait_for_timeout(random.randint(800, 1500))

        # Scroll back up slightly (humans do this)
        await page.evaluate('window.scrollBy(0, -200)')
        await page.wait_for_timeout(random.randint(400, 800))

        # Big scroll down
        await page.evaluate('window.scrollBy(0, 1000)')
        await page.wait_for_timeout(random.randint(1500, 2500))

    async def force_load(self, page):
        """Aggressive methods to force more content loading."""
        # Press End key
        await page.keyboard.press('End')
        await page.wait_for_timeout(2000)

        # Scroll to absolute bottom
        await page.evaluate('window.scrollTo(0, document.body.scrollHeight)')
        await page.wait_for_timeout(3000)

        # Up then down
        await page.evaluate('window.scrollBy(0, -800)')
        await page.wait_for_timeout(1000)
        await page.evaluate('window.scrollBy(0, 1500)')
        await page.wait_for_timeout(2000)

    async def click_last_user(self, page, cells):
        """Click last visible user to trigger loading (last resort)."""
        if not cells:
            return
        try:
            last = cells[-1]
            # Click the user's profile link
            link = await last.query_selector('a[href^="/"]')
            if link:
                await link.hover()
                await page.wait_for_timeout(1000)
                await link.click()
                await page.wait_for_timeout(2000)
                await page.go_back()
                await page.wait_for_timeout(3000)
        except:
            pass

    async def parse_user_cell(self, cell) -> dict:
        try:
            img = await cell.query_selector('img[src*="profile_images"]')
            profile_pic = await img.get_attribute('src') if img else None

            display_name = None
            username = None
            bio = None
            follow_status = None

            all_spans = await cell.query_selector_all('span')
            texts = []
            for span in all_spans:
                text = await span.inner_text()
                text = text.strip()
                if text and text not in texts:
                    texts.append(text)

            for t in texts:
                if t.startswith('@') and len(t) > 1:
                    username = t
                    break

            name_candidates = []
            for t in texts:
                if t and not t.startswith('@') and t not in ['Follows you', 'Following', 'Follow back', 'Follow']:
                    name_candidates.append(t)

            if name_candidates:
                display_name = name_candidates[0]

            if not display_name:
                name_elem = await cell.query_selector('div[data-testid="UserCell"] a div > span')
                if name_elem:
                    display_name = await name_elem.inner_text()
                    if display_name:
                        display_name = display_name.strip()

            if not bio and len(texts) > 2:
                for i, t in enumerate(texts):
                    if t == username and i + 1 < len(texts):
                        next_text = texts[i + 1]
                        if next_text not in ['Follows you', 'Following', 'Follow back', 'Follow']:
                            bio = next_text
                            break

            buttons = await cell.query_selector_all('button')
            for btn in buttons:
                btn_text = await btn.inner_text()
                btn_text = btn_text.strip()
                if btn_text in ['Following', 'Follow back', 'Follow', 'Follows you']:
                    follow_status = btn_text
                    break

            if not follow_status:
                for t in texts:
                    if t == 'Follows you':
                        follow_status = 'Follows you'
                        break

            return {
                'username': f"@{username.lstrip('@')}" if username else None,
                'display_name': display_name,
                'profile_picture': profile_pic,
                'bio': bio,
                'follow_status': follow_status,
                'scraped_at': datetime.utcnow().isoformat()
            }
        except Exception as e:
            return None

    async def scrape_followers(self, target_username: str, mode: str = 'followers', max_items: int = None):
        url = f"https://x.com/{target_username}/{mode}"

        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=False,
                args=[
                    '--disable-blink-features=AutomationControlled',
                    '--disable-web-security',
                    '--disable-features=IsolateOrigins,site-per-process'
                ]
            )

            context = await browser.new_context(
                viewport={'width': 1280, 'height': 900},
                user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
            )

            page = await context.new_page()

            cookies_loaded = await self.load_cookies(page)

            if not cookies_loaded:
                await self.login_and_save_cookies(page)
            else:
                await page.goto('https://x.com/home', wait_until='domcontentloaded', timeout=15000)
                await page.wait_for_timeout(2000)

                if await page.query_selector('a[href="/compose/tweet"]') or await page.query_selector('text=/Home/i'):
                    print("✅ Already logged in!")
                else:
                    print("⚠️ Cookies expired. Logging in again...")
                    await self.login_and_save_cookies(page)

            print(f"\n🌐 Navigating to {url}...")
            await page.goto(url, wait_until='domcontentloaded', timeout=30000)
            await page.wait_for_timeout(5000)

            if await page.query_selector('text=/Sign in to X/i') or await page.query_selector('text=/Log in/i'):
                print("❌ Still not logged in.")
                await browser.close()
                return []

            print("✅ Page loaded! Starting scrape...\n")
            print("💡 TIP: Keep the browser window in focus for best results\n")

            scraped_count = 0
            seen_usernames = set()
            consecutive_no_new = 0
            max_consecutive_no_new = 15
            prev_total = 0
            stall_count = 0

            while True:
                if max_items and scraped_count >= max_items:
                    print(f"\n✅ Reached max_items limit ({max_items}).")
                    break

                cells = await page.query_selector_all('div[data-testid="cellInnerDiv"]')

                new_items = 0
                for cell in cells:
                    user_data = await self.parse_user_cell(cell)

                    if user_data and user_data['username'] and user_data['username'] not in seen_usernames:
                        seen_usernames.add(user_data['username'])
                        self.data.append(user_data)
                        scraped_count += 1
                        new_items += 1
                        print(f"[{scraped_count}] {user_data['username']} — {user_data['display_name'] or 'N/A'}")

                        if max_items and scraped_count >= max_items:
                            break

                print(f"  New: {new_items} | Total: {scraped_count} | Empty streak: {consecutive_no_new}")

                if new_items == 0:
                    consecutive_no_new += 1

                    # Escalating scroll strategies
                    if consecutive_no_new <= 5:
                        await self.human_scroll(page)
                    elif consecutive_no_new <= 10:
                        await self.force_load(page)
                    else:
                        await self.click_last_user(page, cells)
                        await page.evaluate('window.scrollTo(0, document.body.scrollHeight + 2000)')
                        await page.wait_for_timeout(4000)

                    # Check if truly stuck
                    if scraped_count == prev_total:
                        stall_count += 1
                        if stall_count >= 5 and consecutive_no_new >= max_consecutive_no_new:
                            print("\n⏹️ Reached end of list.")
                            break
                    else:
                        stall_count = 0

                    prev_total = scraped_count
                else:
                    consecutive_no_new = 0
                    stall_count = 0
                    prev_total = scraped_count
                    await self.human_scroll(page)

            await browser.close()

        print(f"\n✅ Scraped {len(self.data)} users total.")
        return self.data

    def save_to_json(self, filename: str = 'twitter_followers.json'):
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
        print(f"💾 Saved to {filename}")

    def save_to_csv(self, filename: str = 'twitter_followers.csv'):
        if not self.data:
            print("⚠️ No data to save.")
            return
        keys = ['username', 'display_name', 'profile_picture', 'bio', 'follow_status', 'scraped_at']
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.data)
        print(f"💾 Saved to {filename}")


async def main():
    scraper = TwitterFollowerScraper(cookies_file='twitter_cookies.json')

    await scraper.scrape_followers(
        target_username='parthak',
        mode='followers'
    )

    scraper.save_to_json('my_followers.json')
    scraper.save_to_csv('my_followers.csv')


if __name__ == '__main__':
    asyncio.run(main())

# %%
