import os
import joblib
import pandas as pd
import numpy as np
from utils import format_price_pkr, find_similar_properties, engineer_features_dict

MODEL_PATH = os.path.join("models", "best_model.pkl")
LOC_AVG_PATH = os.path.join("models", "location_avg_price.pkl")
DATA_PATH = os.path.join("data", "processed_zameen_lahore.csv")


def load_model():
    if not os.path.exists(MODEL_PATH):
        print("Model file not found. Please run train_models.py first.")
        return None
    return joblib.load(MODEL_PATH)


def load_location_averages():
    if not os.path.exists(LOC_AVG_PATH):
        print("Location averages mapping file not found. Please run train_models.py first.")
        return {}
    return joblib.load(LOC_AVG_PATH)


def load_dataset():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    return None


def get_number_input(message, default_value=0.0):
    while True:
        user_input = input(message).strip()
        if user_input == "":
            return float(default_value)
        try:
            return float(user_input)
        except ValueError:
            print("Please enter a valid number.")


def get_user_input():
    print("\nEnter property details to estimate price:\n")
    area_marla = get_number_input("Area in Marla: ")
    bedrooms = get_number_input("Number of bedrooms: ")
    bathrooms = get_number_input("Number of bathrooms: ")
    location = input("Location (e.g. DHA Defence, Raiwind): ").strip() or "Unknown"
    property_type = input("Property type [House]: ").strip() or "House"
    built_in_year = get_number_input("Built in year [2020]: ", 2020)
    parking_space = get_number_input("Parking space [0]: ", 0)
    servant_quarters = get_number_input("Servant quarters [0]: ", 0)
    store_rooms = get_number_input("Store rooms [0]: ", 0)
    kitchens = get_number_input("Kitchens [1]: ", 1)
    drawing_rooms = get_number_input("Drawing rooms [1]: ", 1)

    return {
        "area_marla": area_marla,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "location": location,
        "property_type": property_type,
        "built_in_year": built_in_year,
        "parking_space": parking_space,
        "servant_quarters": servant_quarters,
        "store_rooms": store_rooms,
        "kitchens": kitchens,
        "drawing_rooms": drawing_rooms,
    }


def predict_price():
    try:
        model = load_model()
        if model is None:
            return

        loc_averages = load_location_averages()
        df = load_dataset()

        raw_user_input = get_user_input()
        
        # 1. Feature engineering on the raw dictionary inputs
        engineered_dict = engineer_features_dict(raw_user_input, loc_averages)
        
        # Convert to DataFrame
        user_df = pd.DataFrame([engineered_dict])
        
        # 2. Get predictions (predictions are on the log scale)
        predicted_log_price = model.predict(user_df)[0]
        # Revert log transformation to PKR Rupees
        predicted_price = int(round(float(np.expm1(predicted_log_price))))
        
        # Display the result
        print("\n" + "=" * 40)
        print(" ESTIMATED HOUSE PRICE")
        print("-" * 40)
        print(f" Raw Rupees: Rs. {predicted_price:,}")
        print(f" Format PKR: {format_price_pkr(predicted_price)}")
        print("=" * 40 + "\n")
        
        # 3. Find and display similar properties in dataset
        if df is not None:
            similar = find_similar_properties(df, raw_user_input, n=3)
            if similar:
                print("Similar properties in dataset:")
                for i, prop in enumerate(similar, 1):
                    print(f"  {i}. {prop['location']} | {prop['area_marla']} Marla | {int(prop['bedrooms'])} beds | {prop['formatted_price']}")
                print()
            else:
                print("No similar properties found in the dataset.\n")
                
    except Exception as e:
        print(f"Prediction failed. Error details: {e}")
        print("Please check the inputs and try again.")


if __name__ == "__main__":
    predict_price()
