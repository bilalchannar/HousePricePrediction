import os
import re

import pandas as pd


RAW_CSV = os.path.join("data", "raw_zameen_lahore.csv")
PROCESSED_CSV = os.path.join("data", "processed_zameen_lahore.csv")
CSV_COLUMNS = [
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


def clean_text(text):
    if pd.isna(text):
        return None
    text = re.sub(r"\s+", " ", str(text)).strip()
    return text or None


def normalize_location(loc_name):
    if not loc_name:
        return "Unknown"
    loc = str(loc_name).strip()
    
    lowered = loc.lower()
    if "dha defence" in lowered or "dha phase" in lowered or "dha 9 town" in lowered or "dha 11 r" in lowered:
        return "DHA Defence"
    if "bahria town" in lowered:
        return "Bahria Town"
    if "bahria orchard" in lowered:
        return "Bahria Orchard"
    if "lake city" in lowered:
        return "Lake City"
    if "johar town" in lowered:
        return "Johar Town"
    if "wapda town" in lowered:
        return "Wapda Town"
    if "valencia" in lowered:
        return "Valencia Town"
    if "al rehman" in lowered:
        return "Al Rehman Garden"
    if "gt road" in lowered or "hafeez garden" in lowered or "ahmad garden" in lowered:
        return "GT Road Area"
    if "sabzazar" in lowered:
        return "Sabzazar Scheme"
    if "allama iqbal" in lowered or "iqbal town" in lowered:
        return "Iqbal Town"
    if "central park" in lowered:
        return "Central Park Scheme"
    if "formanites" in lowered:
        return "Formanites Scheme"
    if "bankers" in lowered:
        return "Bankers Town"
    if "park view" in lowered:
        return "Park View City"
        
    return loc.title()


def clean_price(price_text):
    if pd.isna(price_text):
        return None

    text = clean_text(price_text)
    if not text:
        return None

    text = text.replace(",", "")
    lowered = text.lower()

    # reject text that is only an area unit
    if re.fullmatch(r"\d+\s*(kanal|marla|sq\.?\s*ft\.?|square feet)", lowered):
        return None

    if not re.search(r"\d", text):
        return None

    total = 0
    found = False

    crore_parts = re.findall(r"(\d+(?:\.\d+)?)\s*crore", lowered)
    for part in crore_parts:
        total += float(part) * 10000000
        found = True

    lakh_parts = re.findall(r"(\d+(?:\.\d+)?)\s*lakh", lowered)
    for part in lakh_parts:
        total += float(part) * 100000
        found = True

    if not found:
        numeric_parts = re.findall(r"\d{5,}", text)
        if numeric_parts:
            total = float(numeric_parts[0])
            found = True

    if found:
        return int(round(total))

    plain_number = re.search(r"\d+(?:\.\d+)?", text)
    if plain_number:
        value = float(plain_number.group())
        if value >= 100000:
            return int(round(value))

    return None


def clean_area(row):
    area_value = row.get("area")
    area_unit = clean_text(row.get("area_unit"))

    if pd.isna(area_value):
        return None

    try:
        area_value = float(area_value)
    except (TypeError, ValueError):
        return None

    if not area_unit:
        return area_value

    unit = area_unit.lower()
    if "kanal" in unit:
        return area_value * 20
    if "marla" in unit:
        return area_value
    if "sq" in unit or "square" in unit:
        if "ft" in unit or "feet" in unit:
            return round(area_value / 272.25, 2)
        if "yd" in unit or "yard" in unit:
            return round(area_value / 30.25, 2)
        if "m" in unit or "meter" in unit:
            return round(area_value / 25.2929, 2)

    return area_value


def preprocess_data():
    if not os.path.exists(RAW_CSV):
        print(f"Raw file not found: {RAW_CSV}")
        return

    df = pd.read_csv(RAW_CSV)
    print(f"Raw dataset shape: {df.shape}")

    if "url" in df.columns:
        df = df.drop_duplicates(subset="url")
    df = df.drop_duplicates()
    print(f"Shape after removing duplicates: {df.shape}")

    missing_before = df.isna().sum()
    print("Missing values before cleaning:")
    print(missing_before.to_string())

    for col in ["city", "location", "property_type"]:
        if col in df.columns:
            df[col] = df[col].apply(clean_text)

    if "price" in df.columns:
        df["price"] = df["price"].apply(clean_price)

    if "area" in df.columns and "area_unit" in df.columns:
        df["area_marla"] = df.apply(clean_area, axis=1)
    else:
        df["area_marla"] = None

    numeric_cols = [
        "bedrooms",
        "bathrooms",
        "built_in_year",
        "parking_space",
        "servant_quarters",
        "store_rooms",
        "kitchens",
        "drawing_rooms",
        "area_marla",
        "price",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "price" in df.columns and "area_marla" in df.columns:
        df = df.dropna(subset=["price", "area_marla"])

    if "price" in df.columns:
        df = df[df["price"] > 0]
    if "area_marla" in df.columns:
        df = df[df["area_marla"] > 0]
    if "bedrooms" in df.columns:
        df = df[df["bedrooms"].between(0, 20, inclusive="both") | df["bedrooms"].isna()]
    if "bathrooms" in df.columns:
        df = df[df["bathrooms"].between(0, 20, inclusive="both") | df["bathrooms"].isna()]

    if "bedrooms" in df.columns:
        median_bedrooms = df["bedrooms"].median()
        if pd.isna(median_bedrooms):
            median_bedrooms = 0
        df["bedrooms"] = df["bedrooms"].fillna(median_bedrooms)

    if "bathrooms" in df.columns:
        median_bathrooms = df["bathrooms"].median()
        if pd.isna(median_bathrooms):
            median_bathrooms = 0
        df["bathrooms"] = df["bathrooms"].fillna(median_bathrooms)

    if "built_in_year" in df.columns:
        # Filter out clearly invalid years (outside 1900-2026) by setting them to None
        df.loc[(df["built_in_year"] < 1900) | (df["built_in_year"] > 2026), "built_in_year"] = None
        median_year = df["built_in_year"].median()
        if pd.isna(median_year) or median_year < 1900:
            median_year = 2020  # Reasonable default year
        df["built_in_year"] = df["built_in_year"].fillna(median_year)

    for col in ["parking_space", "servant_quarters", "store_rooms", "kitchens", "drawing_rooms"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    if "location" in df.columns:
        df["location"] = df["location"].fillna("Unknown").apply(clean_text)
        # Strip common scraped navigation prefixes from location names
        junk_prefix = "GUIDES BLOG MAPS TOOLS MORE Add Property BUY HOMES PLOTS COMMERCIAL RENT AGENTS NEW PROJECTS ZameenLahore Houses"
        df["location"] = df["location"].apply(
            lambda x: x[len(junk_prefix):].strip() if x and x.startswith(junk_prefix) else (
                x.split("ZameenLahore Houses")[-1].strip() if x and "ZameenLahore Houses" in x else x
            )
        )
    else:
        df["location"] = "Unknown"
    if "property_type" in df.columns:
        df["property_type"] = df["property_type"].fillna("House").apply(clean_text)
    else:
        df["property_type"] = "House"

    if "city" in df.columns:
        df["city"] = df["city"].fillna("Lahore").apply(clean_text)
    else:
        df["city"] = "Lahore"

    # 1. Normalize Location Names
    if "location" in df.columns:
        df["location"] = df["location"].apply(normalize_location)

    # 2. Group Rare Locations (counts < 10)
    if "location" in df.columns:
        location_counts = df["location"].value_counts()
        rare_locations = location_counts[location_counts < 10].index
        df["location"] = df["location"].replace(rare_locations, "Other")

    # 3. Filter Outliers
    if "price" in df.columns:
        df = df[df["price"].between(500000, 500000000)]
    if "area_marla" in df.columns:
        df = df[df["area_marla"].between(1, 100)]
    if "bedrooms" in df.columns:
        df = df[df["bedrooms"].between(1, 15)]
    if "bathrooms" in df.columns:
        df = df[df["bathrooms"].between(1, 15)]

    # IQR outlier filtering for price per marla
    if "price" in df.columns and "area_marla" in df.columns and len(df) > 10:
        df["price_per_marla"] = df["price"] / df["area_marla"]
        Q1 = df["price_per_marla"].quantile(0.25)
        Q3 = df["price_per_marla"].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[df["price_per_marla"].between(lower_bound, upper_bound)]
        df = df.drop(columns=["price_per_marla"])

    # 4. Feature Engineering
    if all(col in df.columns for col in ["built_in_year", "bedrooms", "bathrooms", "kitchens", "store_rooms", "servant_quarters", "parking_space", "drawing_rooms"]):
        df["property_age"] = 2026.0 - df["built_in_year"]
        df["total_rooms"] = df["bedrooms"] + df["bathrooms"] + df["kitchens"] + df["store_rooms"]
        df["luxury_score"] = df["servant_quarters"] + df["parking_space"] + df["drawing_rooms"]
        df["bedroom_bathroom_ratio"] = df["bathrooms"] / (df["bedrooms"] + 1e-5)
        
        # Location average price
        location_avg = df.groupby("location")["price"].mean().to_dict()
        df["location_avg_price"] = df["location"].map(location_avg)

    missing_after = df.isna().sum()
    print("Missing values after cleaning:")
    print(missing_after.to_string())

    if "price" in df.columns:
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
    if "area_marla" in df.columns:
        df["area_marla"] = pd.to_numeric(df["area_marla"], errors="coerce")

    df = df.dropna(subset=["price", "area_marla"])

    if "url" in df.columns:
        df = df.drop_duplicates(subset="url")

    engineered_cols = ["property_age", "total_rooms", "luxury_score", "bedroom_bathroom_ratio", "location_avg_price"]
    cols_to_keep = CSV_COLUMNS + ["area_marla"] + engineered_cols
    df = df[[c for c in cols_to_keep if c in df.columns]]

    print(f"Shape after cleaning price and area: {df.shape}")

    df.to_csv(PROCESSED_CSV, index=False, encoding="utf-8")
    print(f"Final processed dataset shape: {df.shape}")
    print(f"Saved file path: {PROCESSED_CSV}")


if __name__ == "__main__":
    preprocess_data()