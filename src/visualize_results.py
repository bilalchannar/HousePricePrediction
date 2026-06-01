import os

import pandas as pd
import matplotlib.pyplot as plt


RESULTS_PATH = os.path.join("models", "model_results.csv")
REPORTS_DIR = "results"


def load_results():
    return pd.read_csv(RESULTS_PATH)


def plot_r2_score(df):
    plt.figure(figsize=(8, 5))
    plt.bar(df["model"], df["r2_score"], color="steelblue")
    plt.title("Model vs R2 Score")
    plt.xlabel("Model")
    plt.ylabel("R2 Score")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "r2_score_comparison.png"), dpi=300)
    plt.close()


def plot_rmse(df):
    plt.figure(figsize=(8, 5))
    plt.bar(df["model"], df["rmse"], color="tomato")
    plt.title("Model vs RMSE")
    plt.xlabel("Model")
    plt.ylabel("RMSE")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "rmse_comparison.png"), dpi=300)
    plt.close()


def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    df = load_results()
    plot_r2_score(df)
    plot_rmse(df)
    print("Charts saved successfully.")


if __name__ == "__main__":
    main()
