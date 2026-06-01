import os
import re
import time

import pandas as pd
from selenium import webdriver
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager


OUTPUT_CSV = os.path.join("data", "raw_zameen_lahore.csv")
PAGE_URL = "https://www.zameen.com/Houses_Property/Lahore-1-{page}.html"
COLS = [
    "price",
    "area",
    "area_unit",
    "city",
    "location",
    "property_type",
    "bedrooms",
    "bathrooms",
    "built_in_year",
    "parking_space",
    "servant_quarters",
    "store_rooms",
    "kitchens",
    "drawing_rooms",
    "url",
]


def setup_driver():
    # setup chrome browser
    options = webdriver.ChromeOptions()
    options.add_argument("--start-maximized")
    options.add_argument("--disable-notifications")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    driver.set_page_load_timeout(45)
    return driver


def clean_text(text):
    if text is None:
        return None
    text = re.sub(r"\s+", " ", str(text)).strip()
    return text or None


def extract_number(text):
    if not text:
        return None
    match = re.search(r"\d+", str(text).replace(",", ""))
    if not match:
        return None
    return int(match.group())


def parse_area(area_text):
    if not area_text:
        return None, None

    area_text = clean_text(area_text)
    match = re.match(r"^(\d+(?:\.\d+)?)\s*(.+)$", area_text or "")
    if not match:
        return area_text, None

    value = float(match.group(1))
    if value.is_integer():
        value = int(value)
    return value, clean_text(match.group(2))


def is_valid_property_url(url):
    if not url:
        return False

    invalid_keywords = [
        "twitter.com",
        "facebook.com",
        "mail.google.com",
        "mailto",
        "sharer",
        "intent/tweet",
        "whatsapp",
        "linkedin",
        "pinterest",
        "javascript:",
        "#",
        "houses_property",
    ]

    url = url.strip()
    lowered = url.lower()

    if any(word in lowered for word in invalid_keywords):
        return False
    if "zameen.com/Property/" not in url:
        return False
    if ".html" not in url:
        return False
    return True


def _page_url(page):
    return PAGE_URL.format(page=page)


def _wait_and_scroll(driver):
    WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
    time.sleep(6)

    try:
        height = driver.execute_script("return document.body.scrollHeight")
        steps = 5
        for i in range(1, steps + 1):
            driver.execute_script(f"window.scrollTo(0, {int(height * i / steps)});")
            time.sleep(0.5)
    except Exception:
        pass


def _restart_driver(driver):
    try:
        driver.quit()
    except Exception:
        pass
    return setup_driver()


def extract_listing_urls(driver):
    # check only real property links
    urls = []
    seen = set()
    raw_count = 0

    try:
        anchors = driver.find_elements(By.TAG_NAME, "a")
    except Exception:
        anchors = []

    for anchor in anchors:
        try:
            href = anchor.get_attribute("href")
        except Exception:
            continue

        if not href:
            continue

        raw_count += 1
        if not is_valid_property_url(href):
            continue

        if href not in seen:
            seen.add(href)
            urls.append(href)

    print(f"Total raw anchor URLs found: {raw_count}")
    print(f"Total valid property URLs found: {len(urls)}")
    print(f"First 5 valid property URLs: {urls[:5]}")
    return urls


def _extract_value(text, labels):
    for label in labels:
        pattern = rf"{re.escape(label)}\s*[:\-]?\s*([^|\n\r]+)"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = clean_text(match.group(1))
            if value:
                return value
    return None


def scrape_listing_detail(driver, listing_url):
    data = {
        "property_type": None,
        "built_in_year": None,
        "parking_space": None,
        "servant_quarters": None,
        "store_rooms": None,
        "kitchens": None,
        "drawing_rooms": None,
        "price": None,
        "area": None,
        "area_unit": None,
        "location": None,
    }

    if not is_valid_property_url(listing_url):
        print("Skipping invalid URL:", listing_url)
        data["property_type"] = "House"
        return data, driver

    try:
        driver.get(listing_url)
        _wait_and_scroll(driver)
    except TimeoutException:
        data["property_type"] = "House"
        return data, driver
    except WebDriverException as exc:
        message = str(exc).lower()
        if "no such window" in message or "web view not found" in message or "target window already closed" in message:
            driver = _restart_driver(driver)
        data["property_type"] = "House"
        return data, driver

    try:
        body_text = driver.find_element(By.TAG_NAME, "body").text or ""
    except Exception:
        body_text = ""

    price_match = re.search(r"(?:PKR|Rs\.?)[\s:]*([\d,]+(?:\.\d+)?(?:\s*[A-Za-z]+)?)", body_text, re.I)
    if price_match:
        data["price"] = clean_text(f"PKR {price_match.group(1)}")

    area_match = re.search(r"(\d+(?:\.\d+)?)\s*(Marla|Kanal|Sq\.\s*Ft\.?|Sq\s*Ft|Square Feet)", body_text, re.I)
    if area_match:
        data["area"], data["area_unit"] = parse_area(f"{area_match.group(1)} {area_match.group(2)}")

    beds_match = re.search(r"(\d+)\s*(Beds?|Bedrooms?)", body_text, re.I)
    if beds_match:
        data["bedrooms"] = extract_number(beds_match.group(1))

    baths_match = re.search(r"(\d+)\s*(Baths?|Bathrooms?)", body_text, re.I)
    if baths_match:
        data["bathrooms"] = extract_number(baths_match.group(1))

    loc = _extract_value(body_text, ["Location", "Area", "Address"])
    if loc:
        data["location"] = loc

    data["property_type"] = _extract_value(body_text, ["Property Type"]) or "House"
    year = _extract_value(body_text, ["Built in year", "Built in Year"])
    data["built_in_year"] = extract_number(year)
    data["parking_space"] = extract_number(_extract_value(body_text, ["Parking Spaces", "Parking Space"]))
    data["servant_quarters"] = extract_number(_extract_value(body_text, ["Servant Quarters"]))
    data["store_rooms"] = extract_number(_extract_value(body_text, ["Store Rooms", "Store Room"]))
    data["kitchens"] = extract_number(_extract_value(body_text, ["Kitchens", "Kitchen"]))
    data["drawing_rooms"] = extract_number(_extract_value(body_text, ["Drawing Room", "Drawing Rooms"]))

    if data["property_type"] is None:
        data["property_type"] = "House"

    return data, driver


def save_to_csv(data):
    # save progress after some records
    _ensure_folder()
    df = pd.DataFrame(data)
    for col in COLS:
        if col not in df.columns:
            df[col] = None
    df = df[COLS]
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"Saved {OUTPUT_CSV}")


def _ensure_folder():
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)


def scrape_lahore_properties(max_records=400):
    print("Starting scraper...")
    _ensure_folder()

    driver = setup_driver()
    data = []
    seen_urls = set()
    page = 1
    max_pages = 50

    try:
        while len(data) < max_records and page <= max_pages:
            print(f"Opening page {page}...")

            try:
                driver.get(_page_url(page))
                _wait_and_scroll(driver)
            except WebDriverException as exc:
                message = str(exc).lower()
                if "no such window" in message or "web view not found" in message or "target window already closed" in message:
                    driver = _restart_driver(driver)
                    continue
                page += 1
                continue

            links = extract_listing_urls(driver)
            valid_links = [link for link in links if is_valid_property_url(link)]
            print(f"Found {len(valid_links)} valid property links.")

            if not valid_links:
                page += 1
                continue

            for listing_url in valid_links:
                if len(data) >= max_records:
                    break
                if listing_url in seen_urls:
                    continue

                seen_urls.add(listing_url)
                print(f"Scraping listing {len(data) + 1}/{max_records}...")

                listing_data, driver = scrape_listing_detail(driver, listing_url)
                listing_data["city"] = "Lahore"
                listing_data["url"] = listing_url
                if listing_data["property_type"] is None:
                    listing_data["property_type"] = "House"
                data.append(listing_data)

                if len(data) % 25 == 0:
                    save_to_csv(data)

                time.sleep(1)

            page += 1

    finally:
        save_to_csv(data)
        try:
            driver.quit()
        except Exception:
            pass

    print(f"Total records scraped: {len(data)}")
    return pd.DataFrame(data)


if __name__ == "__main__":
    scrape_lahore_properties(max_records=400)