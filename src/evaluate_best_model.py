import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

DATA_PATH = os.path.join("data", "processed_zameen_lahore.csv")
MODEL_RESULTS_PATH = os.path.join("models", "model_results.csv")
BEST_MODEL_PATH = os.path.join("models", "best_model.pkl")
RESULTS_DIR = "results"

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


def _ensure_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


def load_data():
    return pd.read_csv(DATA_PATH)


def load_model_results():
    return pd.read_csv(MODEL_RESULTS_PATH)


def plot_model_comparison(results_df):
    _ensure_results_dir()
    
    # Filter out Dummy Regressor or keep it to show baseline comparison
    plt.figure(figsize=(8, 4))
    plt.bar(results_df["model"], results_df["r2_score"], color="steelblue")
    plt.title("Model vs R2 Score")
    plt.xlabel("Model")
    plt.ylabel("R2 Score")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "r2_score_comparison.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.bar(results_df["model"], results_df["rmse"], color="tomato")
    plt.title("Model vs RMSE")
    plt.xlabel("Model")
    plt.ylabel("RMSE")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "rmse_comparison.png"), dpi=300)
    plt.close()


def evaluate_model():
    df = load_data().copy()
    df = df[FEATURES + ["price"]].copy()

    for col in FEATURES:
        if col in ["location", "property_type"]:
            df[col] = df[col].fillna("Unknown").astype(str)
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            median_value = df[col].median()
            if pd.isna(median_value):
                median_value = 0
            df[col] = df[col].fillna(median_value)

    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["price"])
    df = df[df["price"] > 0]

    X = df[FEATURES]
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    model = joblib.load(BEST_MODEL_PATH)
    
    # Predictions are in log scale because the model was trained on y_train_log
    predictions_log = model.predict(X_test)
    # Reverse log target transformation to get PKR prices
    predictions = np.expm1(predictions_log)

    metrics = {
        "mae": mean_absolute_error(y_test, predictions),
        "mse": mean_squared_error(y_test, predictions),
        "rmse": np.sqrt(mean_squared_error(y_test, predictions)),
        "r2_score": r2_score(y_test, predictions),
        "mape": mean_absolute_percentage_error(y_test, predictions)
    }

    return y_test, predictions, metrics, model, df


def plot_actual_vs_predicted(y_test, predictions):
    _ensure_results_dir()

    # Log scaling scatter plot since real prices span orders of magnitude
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test / 10000000.0, predictions / 10000000.0, alpha=0.6, color="purple")
    # Draw perfect prediction line
    max_val = max(y_test.max(), predictions.max()) / 10000000.0
    plt.plot([0, max_val], [0, max_val], 'k--', alpha=0.7)
    plt.title("Actual vs Predicted House Prices")
    plt.xlabel("Actual Price (Crore PKR)")
    plt.ylabel("Predicted Price (Crore PKR)")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "actual_vs_predicted.png"), dpi=300)
    plt.close()


def plot_residuals(y_test, predictions):
    _ensure_results_dir()
    residuals = y_test - predictions
    
    plt.figure(figsize=(8, 4))
    plt.hist(residuals / 10000000.0, bins=25, color="teal", edgecolor="black", alpha=0.7)
    plt.axvline(0, color="red", linestyle="--")
    plt.title("Residuals Distribution")
    plt.xlabel("Error (Actual - Predicted in Crore PKR)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "residuals_plot.png"), dpi=300)
    plt.close()


def plot_price_distributions(df):
    _ensure_results_dir()
    
    plt.figure(figsize=(10, 4))
    
    # Original Price Distribution
    plt.subplot(1, 2, 1)
    plt.hist(df["price"] / 10000000.0, bins=20, color="slateblue", edgecolor="black", alpha=0.7)
    plt.title("Original Price Distribution")
    plt.xlabel("Price (Crore PKR)")
    plt.ylabel("Count")
    
    # Log-Transformed Price Distribution
    plt.subplot(1, 2, 2)
    plt.hist(np.log1p(df["price"]), bins=20, color="seagreen", edgecolor="black", alpha=0.7)
    plt.title("Log-Transformed Price Distribution")
    plt.xlabel("Log(Price)")
    plt.ylabel("Count")
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "price_distribution.png"), dpi=300)
    plt.close()


def plot_location_avg_price(df):
    _ensure_results_dir()
    
    # Calculate average price by location
    avg_prices = df.groupby("location")["price"].mean() / 10000000.0
    avg_prices = avg_prices.sort_values(ascending=False).head(10)
    
    plt.figure(figsize=(10, 5))
    plt.barh(avg_prices.index, avg_prices.values, color="goldenrod", edgecolor="black", alpha=0.8)
    plt.title("Top 10 Average House Prices by Location")
    plt.xlabel("Average Price (Crore PKR)")
    plt.ylabel("Location")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "location_price_chart.png"), dpi=300)
    plt.close()


def save_error_by_range(y_test, predictions):
    _ensure_results_dir()
    
    eval_df = pd.DataFrame({
        "actual": y_test,
        "predicted": predictions
    })
    eval_df["abs_error"] = (eval_df["actual"] - eval_df["predicted"]).abs()
    eval_df["percentage_error"] = eval_df["abs_error"] / eval_df["actual"]
    
    # Define ranges in PKR
    ranges = [
        ("Below 1 Crore", eval_df[eval_df["actual"] < 10000000]),
        ("1 to 3 Crore", eval_df[eval_df["actual"].between(10000000, 30000000)]),
        ("3 to 5 Crore", eval_df[eval_df["actual"].between(30000000, 50000000)]),
        ("Above 5 Crore", eval_df[eval_df["actual"] > 50000000])
    ]
    
    breakdown_results = []
    for label, group in ranges:
        if not group.empty:
            breakdown_results.append({
                "price_range": label,
                "count": len(group),
                "mae": group["abs_error"].mean(),
                "mape": group["percentage_error"].mean() * 100
            })
            
    range_df = pd.DataFrame(breakdown_results)
    range_df.to_csv(os.path.join(RESULTS_DIR, "error_by_range.csv"), index=False)


def save_error_file(y_test, predictions):
    _ensure_results_dir()

    error_df = pd.DataFrame({
        "actual_price": y_test.values,
        "predicted_price": predictions,
    })
    error_df["error"] = error_df["actual_price"] - error_df["predicted_price"]
    error_df["absolute_error"] = error_df["error"].abs()
    error_df.to_csv(os.path.join(RESULTS_DIR, "prediction_errors.csv"), index=False, encoding="utf-8")


def plot_feature_importance(model):
    _ensure_results_dir()

    try:
        preprocessor = model.named_steps["preprocessor"]
        reg_model = model.named_steps["model"]
        importances = reg_model.feature_importances_
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        print("Feature importance not available for this model.")
        return

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values(by="importance", ascending=False).head(15)

    plt.figure(figsize=(8, 6))
    plt.barh(importance_df["feature"], importance_df["importance"], color="green")
    plt.title("Top Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "feature_importance.png"), dpi=300)
    plt.close()


def save_summary(df, metrics):
    _ensure_results_dir()

    summary_path = os.path.join(RESULTS_DIR, "dataset_summary.txt")
    text = []
    text.append(f"Total processed rows: {len(df)}")
    text.append(f"Total features used: {len(FEATURES)}")
    text.append(f"Training rows: {int(len(df) * 0.8)}")
    text.append(f"Testing rows: {int(len(df) * 0.2)}")
    text.append(f"MAE (PKR): {metrics['mae']:.2f}")
    text.append(f"MSE (PKR): {metrics['mse']:.2f}")
    text.append(f"RMSE (PKR): {metrics['rmse']:.2f}")
    text.append(f"R2 Score: {metrics['r2_score']:.4f}")
    text.append(f"Average Percentage Error (MAPE): {metrics['mape'] * 100:.2f}%")
    text.append(f"Best model file path: {BEST_MODEL_PATH}")
    text.append("Extra evaluation charts were created for report analysis.")

    with open(summary_path, "w", encoding="utf-8") as file:
        file.write("\n".join(text))


def main():
    _ensure_results_dir()
    results_df = load_model_results()
    plot_model_comparison(results_df)

    y_test, predictions, metrics, model, df = evaluate_model()
    plot_actual_vs_predicted(y_test, predictions)
    plot_residuals(y_test, predictions)
    plot_price_distributions(df)
    plot_location_avg_price(df)
    save_error_by_range(y_test, predictions)
    save_error_file(y_test, predictions)
    plot_feature_importance(model)
    save_summary(df, metrics)

    print("Evaluation completed.")
    print("Files saved in results folder.")


if __name__ == "__main__":
    main()
