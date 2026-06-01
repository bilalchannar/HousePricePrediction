import os
import pandas as pd
import numpy as np

def format_price_pkr(price):
    """Formats a price number into a human-readable Pakistani Rupees (PKR) format (Crores/Lakhs)."""
    try:
        price = float(price)
    except (ValueError, TypeError):
        return str(price)
        
    if price >= 10000000:
        return f"PKR {price / 10000000:.2f} Crore"
    elif price >= 100000:
        return f"PKR {price / 100000:.2f} Lakh"
    else:
        return f"PKR {price:,.0f}"

def find_similar_properties(df, input_dict, n=3):
    """Finds n similar properties in the dataset based on location, area, and bedrooms."""
    if df is None or df.empty:
        return []
        
    # Copy to avoid modifying the original dataframe
    temp_df = df.copy()
    
    # 1. Filter by location if possible, otherwise fallback to entire dataset
    loc = input_dict.get("location", "Unknown")
    same_loc_df = temp_df[temp_df["location"].str.lower() == loc.lower()]
    if not same_loc_df.empty:
        temp_df = same_loc_df
        
    # 2. Calculate a simple distance score
    # Score = |area_diff| / 5 + |beds_diff| * 2 + |baths_diff|
    area = float(input_dict.get("area_marla", 0))
    beds = float(input_dict.get("bedrooms", 0))
    baths = float(input_dict.get("bathrooms", 0))
    
    temp_df["distance"] = (
        (temp_df["area_marla"] - area).abs() / 5.0 +
        (temp_df["bedrooms"] - beds).abs() * 2.0 +
        (temp_df["bathrooms"] - baths).abs()
    )
    
    # Sort by distance and return top n records
    similar_properties = temp_df.sort_values(by="distance").head(n)
    
    results = []
    for _, row in similar_properties.iterrows():
        results.append({
            "location": row["location"],
            "area_marla": row["area_marla"],
            "bedrooms": row["bedrooms"],
            "bathrooms": row["bathrooms"],
            "price": row["price"],
            "formatted_price": format_price_pkr(row["price"])
        })
        
    return results

def engineer_features_dict(input_dict, location_avg_prices):
    """Dynamically engineers features for a single input dictionary during inference."""
    # Read base values
    built_in_year = float(input_dict.get("built_in_year", 0))
    bedrooms = float(input_dict.get("bedrooms", 0))
    bathrooms = float(input_dict.get("bathrooms", 0))
    kitchens = float(input_dict.get("kitchens", 0))
    store_rooms = float(input_dict.get("store_rooms", 0))
    servant_quarters = float(input_dict.get("servant_quarters", 0))
    parking_space = float(input_dict.get("parking_space", 0))
    drawing_rooms = float(input_dict.get("drawing_rooms", 0))
    location = str(input_dict.get("location", "Unknown")).strip()
    
    # Feature calculations
    property_age = 2026.0 - built_in_year if built_in_year > 0 else 6.0 # 6.0 is median age (2026-2020)
    total_rooms = bedrooms + bathrooms + kitchens + store_rooms
    luxury_score = servant_quarters + parking_space + drawing_rooms
    bedroom_bathroom_ratio = bathrooms / (bedrooms + 1e-5)
    
    # Look up location average price
    location_avg_price = location_avg_prices.get(location, location_avg_prices.get("Other", 0.0))
    if pd.isna(location_avg_price) or location_avg_price == 0:
        location_avg_price = location_avg_prices.get("Other", 0.0)
        
    # Return a dict containing all original + engineered features in the correct order
    return {
        "area_marla": float(input_dict.get("area_marla", 0)),
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "location": location,
        "property_type": str(input_dict.get("property_type", "House")).strip(),
        "built_in_year": built_in_year,
        "parking_space": parking_space,
        "servant_quarters": servant_quarters,
        "store_rooms": store_rooms,
        "kitchens": kitchens,
        "drawing_rooms": drawing_rooms,
        # Engineered features
        "property_age": property_age,
        "total_rooms": total_rooms,
        "luxury_score": luxury_score,
        "bedroom_bathroom_ratio": bedroom_bathroom_ratio,
        "location_avg_price": location_avg_price
    }
