import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from src.utils import format_price_pkr, find_similar_properties, engineer_features_dict

# App configuration
st.set_page_config(
    page_title="Lahore House Price Prediction System",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paths
MODEL_PATH = "models/best_model.pkl"
LOC_AVG_PATH = "models/location_avg_price.pkl"
DATA_PATH = "data/processed_zameen_lahore.csv"
RESULTS_DIR = "results"

# Load serialized states
@st.cache_resource
def load_ml_pipeline():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

@st.cache_data
def load_location_avg():
    if os.path.exists(LOC_AVG_PATH):
        return joblib.load(LOC_AVG_PATH)
    return {}

@st.cache_data
def load_dataset():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    return None

pipeline = load_ml_pipeline()
loc_avg_prices = load_location_avg()
dataset = load_dataset()

# Custom premium styling
st.markdown("""
    <style>
        .main-header {
            background: linear-gradient(135deg, #1f4e79 0%, #102a43 100%);
            padding: 2.5rem;
            border-radius: 12px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
        }
        .prediction-card {
            background-color: #e8f5e9;
            border-left: 6px solid #2e7d32;
            padding: 1.8rem;
            border-radius: 8px;
            margin-top: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 15px rgba(46, 125, 50, 0.1);
        }
        .metric-title {
            font-size: 1.1rem;
            font-weight: bold;
            color: #2e7d32;
            margin-bottom: 0.2rem;
        }
        .metric-value {
            font-size: 2.2rem;
            font-weight: 800;
            color: #1b5e20;
            margin-bottom: 0.5rem;
        }
        .metric-sub {
            font-size: 1rem;
            color: #33691e;
            font-weight: 500;
        }
    </style>
""", unsafe_allow_html=True)

# App Header
st.markdown("""
    <div class="main-header">
        <h1>🏠 Lahore House Price Prediction System</h1>
        <p>Enter property details below to estimate the market value using our trained Machine Learning models.</p>
    </div>
""", unsafe_allow_html=True)

if pipeline is None:
    st.error("⚠️ Best model pipeline (`best_model.pkl`) not found. Please run Option 3 (Train models) in `main.py` first.")
    st.stop()

# Populate Location Dropdown options
if loc_avg_prices:
    locations_list = sorted([k for k in loc_avg_prices.keys() if k not in ["Other", "Unknown"]])
    if "Other" not in locations_list:
        locations_list.append("Other")
else:
    locations_list = ["DHA Defence", "Bahria Town", "Johar Town", "Lake City", "GT Road Area", "Sabzazar Scheme", "Other"]

# Sidebar information
with st.sidebar:
    st.markdown("### 📊 Model Insights")
    if os.path.exists("models/model_results.csv"):
        results_df = pd.read_csv("models/model_results.csv")
        st.dataframe(
            results_df[["model", "r2_score", "rmse"]].sort_values(by="r2_score", ascending=False),
            use_container_width=True,
            hide_index=True
        )
    st.info("💡 **Tip**: Unit selections automatically convert area values internally to Marlas before running model calculations.")
    st.markdown("---")
    st.markdown("Developed for Zameen Lahore Property Prediction")

# Input Layout splits
col1, col2 = st.columns([3, 2], gap="large")

with col1:
    st.markdown("### 📝 Enter Property Specifications")
    
    with st.expander("📍 Location & Type", expanded=True):
        sc1, sc2 = st.columns(2)
        with sc1:
            location = st.selectbox("Location / Area", locations_list, index=0)
        with sc2:
            property_type = st.selectbox("Property Type", ["House", "Flat", "Penthouse", "Room", "Upper Portion", "Lower Portion"], index=0)
            
    with st.expander("📐 Area & Dimensions", expanded=True):
        sc1, sc2 = st.columns([3, 2])
        with sc1:
            area_val = st.number_input("Area Value", min_value=0.1, max_value=10000.0, value=10.0, step=0.5)
        with sc2:
            unit = st.selectbox("Area Unit", ["Marla", "Kanal", "Square Feet", "Square Yards"], index=0)
            
    with st.expander("🛏️ Rooms & Spaces", expanded=True):
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            bedrooms = st.slider("Bedrooms", 1, 15, 4)
            kitchens = st.slider("Kitchens", 1, 5, 2)
        with sc2:
            bathrooms = st.slider("Bathrooms", 1, 15, 5)
            drawing_rooms = st.slider("Drawing Rooms", 0, 5, 1)
        with sc3:
            parking_space = st.slider("Parking Space", 0, 5, 2)
            store_rooms = st.slider("Store Rooms", 0, 5, 1)
            
    with st.expander("🛠️ Construction & Extras", expanded=True):
        sc1, sc2 = st.columns(2)
        with sc1:
            built_in_year = st.number_input("Built-in Year (Enter 0 if unknown)", min_value=0, max_value=2026, value=2022, step=1)
        with sc2:
            servant_quarters = st.slider("Servant Quarters", 0, 5, 1)

    # Unit Conversion Logic
    if unit == "Kanal":
        area_marla = area_val * 20.0
    elif unit == "Square Feet":
        area_marla = area_val / 272.25
    elif unit == "Square Yards":
        area_marla = area_val / 30.25
    else: # Marla
        area_marla = area_val

    # Construct input dictionaries
    raw_input_dict = {
        "area_marla": area_marla,
        "bedrooms": float(bedrooms),
        "bathrooms": float(bathrooms),
        "location": location,
        "property_type": property_type,
        "built_in_year": float(built_in_year if built_in_year > 0 else 2020),
        "parking_space": float(parking_space),
        "servant_quarters": float(servant_quarters),
        "store_rooms": float(store_rooms),
        "kitchens": float(kitchens),
        "drawing_rooms": float(drawing_rooms)
    }

with col2:
    st.markdown("### 🎯 Estimated Market Value")
    
    # Feature engineering for prediction
    engineered_dict = engineer_features_dict(raw_input_dict, loc_avg_prices)
    input_df = pd.DataFrame([engineered_dict])
    
    try:
        # Run prediction
        pred_log = pipeline.predict(input_df)[0]
        prediction = int(round(float(np.expm1(pred_log))))
        
        # Load best model RMSE to calculate ranges
        rmse = 45000000.0
        if os.path.exists("models/model_results.csv"):
            rdf = pd.read_csv("models/model_results.csv")
            rmse = float(rdf.sort_values(by="r2_score", ascending=False).iloc[0]["rmse"])
            
        lower_price = max(500000, prediction - rmse)
        upper_price = prediction + rmse
        
        # Confidence logic
        if location != "Other":
            confidence = "High"
            conf_color = "green"
        else:
            confidence = "Medium (Fallback to Location Average)"
            conf_color = "orange"
            
        # Price per Marla
        price_per_marla = prediction / area_marla if area_marla > 0 else 0
        
        # Display Prediction Card
        st.markdown(f"""
            <div class="prediction-card">
                <div class="metric-title">Estimated House Value</div>
                <div class="metric-value">{format_price_pkr(prediction)}</div>
                <div class="metric-sub">Expected Range: {format_price_pkr(lower_price)} - {format_price_pkr(upper_price)}</div>
                <hr style="border-top: 1px solid #c8e6c9; margin: 0.8rem 0;">
                <div style="font-size: 0.95rem; color: #37474f;">
                    <b>Confidence:</b> <span style="color: {conf_color}; font-weight: bold;">{confidence}</span><br>
                    <b>Avg. Price per Marla:</b> Rs. {int(price_per_marla):,}
                </div>
            </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Prediction crashed. Error: {e}")
        
    # Similar Properties Display
    if dataset is not None:
        st.markdown("#### 🔍 Similar Properties in Dataset")
        similar = find_similar_properties(dataset, raw_input_dict, n=3)
        if similar:
            for s in similar:
                st.markdown(f"""
                    * **{s['location']}** | {s['area_marla']} Marla | {int(s['bedrooms'])} Bed | {int(s['bathrooms'])} Bath  
                      **Price**: {s['formatted_price']}
                """)
        else:
            st.write("No similar properties found.")

# Data Visualizations Section
st.markdown("---")
st.markdown("### 📊 Interactive Visual Analytics")

tab1, tab2, tab3 = st.tabs(["🎯 Feature Importances", "📉 Error & Residual Plots", "💵 Location Prices"])

with tab1:
    fig_path = os.path.join(RESULTS_DIR, "feature_importance.png")
    if os.path.exists(fig_path):
        st.image(fig_path, caption="CatBoost Top Feature Importances (PKR Scale)", use_container_width=True)
    else:
        st.info("Feature importance plot not found. Run evaluations script to generate.")

with tab2:
    sc1, sc2 = st.columns(2)
    with sc1:
        fig_path = os.path.join(RESULTS_DIR, "residuals_plot.png")
        if os.path.exists(fig_path):
            st.image(fig_path, caption="Residual Errors distribution", use_container_width=True)
        else:
            st.info("Residual plot not found.")
    with sc2:
        fig_path = os.path.join(RESULTS_DIR, "price_distribution.png")
        if os.path.exists(fig_path):
            st.image(fig_path, caption="Log Price vs Target Price Distributions", use_container_width=True)
        else:
            st.info("Price distribution chart not found.")

with tab3:
    fig_path = os.path.join(RESULTS_DIR, "location_price_chart.png")
    if os.path.exists(fig_path):
        st.image(fig_path, caption="Lahore Location Pricing Comparison", use_container_width=True)
    else:
        st.info("Location average price chart not found.")
