import random
import time
from urllib.parse import urlparse

from playwright.sync_api import sync_playwright

BLOCKED_DOMAINS = (
    "google.com",
    "about.google",
    "youtube.com",
    "youtu.be",
    "facebook.com",
    "instagram.com",
    "x.com",
    "twitter.com",
)


def stealth_sync(page):
    page.add_init_script("""
        Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
        window.chrome = { runtime: {} };
        Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
    """)


def unique_preserve_order(items):
    seen = set()
    out = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def scrape_news_content(context, url, results_array):
    page = None
    try:
        page = context.new_page()
        stealth_sync(page)
        print(f"   >> Scraping: {url}")
        page.goto(url, timeout=45000, wait_until="domcontentloaded")
        time.sleep(random.uniform(2, 3))

        heading = (
            page.get_by_role("heading", level=1).first.inner_text()
            if page.get_by_role("heading", level=1).count() > 0
            else "No Heading"
        )
        paragraphs = page.locator("p").all_inner_texts()
        full_body = "\n\n".join([p.strip() for p in paragraphs if len(p.strip()) > 60])

        if full_body:
            results_array.append({
                "source": url,
                "heading": heading,
                "content": full_body
            })
            print("   >> Added to result array.")
    except Exception as e:
        print(f"   >> Error: {str(e)[:80]}")
    finally:
        if page is not None:
            page.close()


def run(max_topics: int = 3, max_links_per_topic: int = 3, headless: bool = False):
    results = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/121.0.0.0 Safari/537.36"
        )
        page = context.new_page()
        stealth_sync(page)

        print("Navigating to Google Trends...")
        page.goto("https://trends.google.com/trending?geo=IN&category=9")
        page.get_by_role("row").nth(1).wait_for(timeout=30000)

        row_count = page.get_by_role("row").count()
        available_topics = max(0, row_count - 1)
        topics_to_scrape = min(max(1, max_topics), available_topics)

        for i in range(topics_to_scrape):
            try:
                current_row = page.get_by_role("row").nth(i + 1)
                title_cell = current_row.get_by_role("gridcell").nth(1)
                topic_name = title_cell.inner_text().split("\n")[0].strip()
                print(f"\n--- Topic #{i + 1}: {topic_name} ---")

                title_cell.click()
                time.sleep(2)

                all_links_locator = page.get_by_role("link").filter(has_text=None)
                if all_links_locator.count() > 0:
                    all_links_locator.last.scroll_into_view_if_needed()
                    time.sleep(1)

                links = page.get_by_role("link").all()
                valid_urls = []

                for link in links:
                    url = link.get_attribute("href")
                    if not url or "http" not in url:
                        continue

                    netloc = urlparse(url).netloc.lower()
                    if any(domain in netloc for domain in BLOCKED_DOMAINS):
                        continue

                    valid_urls.append(url)

                valid_urls = unique_preserve_order(valid_urls)
                limited_urls = valid_urls[: max(1, max_links_per_topic)]
                print(f"   Found {len(limited_urls)} articles.")

                for url in limited_urls:
                    scrape_news_content(context, url, results)

                title_cell.click()
                time.sleep(1)
            except Exception as e:
                print(f"   Skipping Topic {i + 1}: {e}")

        browser.close()

    return results


if __name__ == "__main__":
    data = run()
    print(f"\nCollected {len(data)} articles.")
