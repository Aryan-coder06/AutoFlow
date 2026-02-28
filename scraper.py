import time
import random
from playwright.sync_api import sync_playwright

def stealth_sync(page):
    page.add_init_script("""
        Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
        window.chrome = { runtime: {} };
        Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
    """)

def scrape_news_content(context, url, results_array):
    try:
        page = context.new_page()
        stealth_sync(page)
        print(f"   >> Scraping: {url}")
        page.goto(url, timeout=45000, wait_until="domcontentloaded")
        time.sleep(random.uniform(2, 3))

        # Extract Heading
        heading = page.get_by_role("heading", level=1).first.inner_text() \
            if page.get_by_role("heading", level=1).count() > 0 else "No Heading"
        
        # Extract Content
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
        print(f"   >> Error: {str(e)[:50]}")
    finally:
        page.close()

def run():
    results = []   # <-- ARRAY OF OBJECTS

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/121.0.0.0 Safari/537.36"
        )
        page = context.new_page()
        stealth_sync(page)

        print("Navigating to Google Trends...")
        page.goto("https://trends.google.com/trending?geo=IN&category=9")

        # Wait for rows
        page.get_by_role("row").nth(1).wait_for()

        # Top 3 topics
        for i in range(1):
            try:
                current_row = page.get_by_role("row").nth(i + 1)
                title_cell = current_row.get_by_role("gridcell").nth(1)
                topic_name = title_cell.inner_text().split("\n")[0]

                print(f"\n--- Topic #{i+1}: {topic_name} ---")

                title_cell.click()
                time.sleep(2)

                # Scroll for lazy loading 
                all_links_locator = page.get_by_role("link").filter(has_text=None)
                if all_links_locator.count() > 0:
                    all_links_locator.last.scroll_into_view_if_needed()
                    time.sleep(1)

                # Extract links
                links = page.get_by_role("link").all()
                valid_urls = []

                for link in links:
                    url = link.get_attribute("href")
                    if url and "http" in url and "google.com" not in url:
                        valid_urls.append(url)

                valid_urls = list(set(valid_urls))
                print(f"   Found {len(valid_urls)} articles.")

                for url in valid_urls:
                    scrape_news_content(context, url, results)

                # Close expansion
                title_cell.click()
                time.sleep(1)

            except Exception as e:
                print(f"   Skipping Topic {i+1}: {e}")

        browser.close()

    return results   # <-- RETURN ARRAY


if __name__ == "__main__":
    data = run()
    print(f"\nCollected {len(data)} articles.")