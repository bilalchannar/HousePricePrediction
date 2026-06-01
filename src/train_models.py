import os
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error, median_absolute_error

DATA_PATH = os.path.join("data", "processed_zameen_lahore.csv")
MODEL_DIR = "models"
RESULTS_PATH = os.path.join(MODEL_DIR, "model_results.csv")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "preprocessor.pkl")

FEATURES = [
    "area_marla",
    "bedrooms",
    "bathrooms",
    "location",
    "property_type",
    "built_in_year",
    "parking_space",
    "servant_quarters",
    "store_rooms",
    "kitchens",
    "drawing_rooms",
    "property_age",
    "total_rooms",
    "luxury_score",
    "bedroom_bathroom_ratio",
    "location_avg_price",
]

NUMERIC_COLS = [
    "area_marla",
    "bedrooms",
    "bathrooms",
    "built_in_year",
    "parking_space",
    "servant_quarters",
    "store_rooms",
    "kitchens",
    "drawing_rooms",
    "property_age",
    "total_rooms",
    "luxury_score",
    "bedroom_bathroom_ratio",
    "location_avg_price",
]

CATEGORICAL_COLS = ["location", "property_type"]


def load_data():
    return pd.read_csv(DATA_PATH)


def _make_preprocessor():
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    return ColumnTransformer(
        transformers=[
            ("cat", encoder, CATEGORICAL_COLS),
            ("num", "passthrough", NUMERIC_COLS),
        ]
    )


def train_models():
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = load_data()
    print(f"Dataset shape: {df.shape}")

    needed_cols = FEATURES + ["price"]
    df = df[needed_cols].copy()

    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    df["location"] = df["location"].fillna("Unknown").astype(str).str.strip()
    df["property_type"] = df["property_type"].fillna("House").astype(str).str.strip()

    for col in NUMERIC_COLS:
        if col in df.columns:
            median_value = df[col].median()
            if pd.isna(median_value):
                median_value = 0
            df[col] = df[col].fillna(median_value)

    df = df.dropna(subset=["price"])
    df = df[df["price"] > 0]

    X = df[FEATURES].copy()
    y = df["price"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    print(f"Training rows: {X_train.shape[0]}")
    print(f"Testing rows: {X_test.shape[0]}")

    # Calculate and save location average price map from training set to avoid data leakage
    location_avg_map = y_train.groupby(X_train["location"]).mean().to_dict()
    # Add default fallbacks for unrepresented/rare location names
    overall_mean_price = y_train.mean()
    location_avg_map["Other"] = overall_mean_price
    location_avg_map["Unknown"] = overall_mean_price
    
    # Save the mapping file
    joblib.dump(location_avg_map, os.path.join(MODEL_DIR, "location_avg_price.pkl"))
    print("Saved models/location_avg_price.pkl")

    # Fit preprocessor separately and serialize it
    preprocessor = _make_preprocessor()
    preprocessor.fit(X_train)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)

    # Log target transformation for training
    y_train_log = np.log1p(y_train)

    models = {
        "Dummy Baseline": DummyRegressor(strategy="median"),
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "XGBoost": XGBRegressor(
            n_estimators=100,
            random_state=42,
            objective="reg:squarederror",
        ),
        "CatBoost": CatBoostRegressor(
            iterations=500,  # Increased for better convergence
            depth=6,
            learning_rate=0.05,
            random_state=42,
            verbose=0,
        ),
    }

    results = []
    best_r2 = -np.inf
    best_name = None
    best_pipeline = None

    for name, model in models.items():
        print(f"Training {name}...")

        pipeline = Pipeline([
            ("preprocessor", _make_preprocessor()),
            ("model", model),
        ])

        # Train on log target
        pipeline.fit(X_train, y_train_log)
        
        # Cross Validation Score on Log Targets
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(pipeline, X_train, y_train_log, cv=cv, scoring="r2")
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()

        # Predict on Test Set
        preds_log = pipeline.predict(X_test)
        # Convert predictions back to PKR scale before calculating performance metrics
        preds = np.expm1(preds_log)

        # Calculate PKR-scale metrics
        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, preds)
        mape = mean_absolute_percentage_error(y_test, preds)
        med_ae = median_absolute_error(y_test, preds)

        print(f"{name} -> MAE: {mae:.2f}, R2: {r2:.4f}, CV-R2: {cv_mean:.4f} (+/- {cv_std:.4f})")

        results.append({
            "model": name,
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "r2_score": r2,
            "mape": mape,
            "median_absolute_error": med_ae,
            "cv_mean_r2": cv_mean,
            "cv_std_r2": cv_std
        })

        if r2 > best_r2:
            best_r2 = r2
            best_name = name
            best_pipeline = pipeline

    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_PATH, index=False, encoding="utf-8")

    # Save the best pipeline (preprocessor + model)
    joblib.dump(best_pipeline, BEST_MODEL_PATH)

    print(f"\nBest model name: {best_name} (Test R2: {best_r2:.4f})")
    print(f"Saved file paths: {RESULTS_PATH}, {BEST_MODEL_PATH}, {PREPROCESSOR_PATH}")


if __name__ == "__main__":
    train_models()
