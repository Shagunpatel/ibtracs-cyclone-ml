# visualize.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from train import DATA_PATH, MODEL_PATH, load_and_preprocess

FIG_DIR = "figures"

def plot_wind_distribution(df: pd.DataFrame):
    plt.figure()
    df["USA_WIND"].hist(bins=30)
    plt.xlabel("Wind speed (knots)")
    plt.ylabel("Count")
    plt.title("Distribution of USA_WIND")
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "wind_distribution.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved {out_path}")

def plot_feature_importance():
    # Use the same preprocessed data and trained model
    X_train, X_test, y_train, y_test = load_and_preprocess()
    model = joblib.load(MODEL_PATH)

    importances = model.feature_importances_
    feature_names = X_train.columns

    plt.figure()
    plt.bar(range(len(importances)), importances)
    plt.xticks(range(len(importances)), feature_names, rotation=45, ha="right")
    plt.ylabel("Importance")
    plt.title("Random Forest Feature Importances")
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "feature_importance.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved {out_path}")

def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    # Load raw data for distribution plot
    print(f"Loading data from {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH, low_memory=False)

    # Convert wind speed column to numeric (removes strings like "knots" or "na")
    df["USA_WIND"] = pd.to_numeric(df["USA_WIND"], errors="coerce")

    # Drop rows where wind speed is missing or invalid
    df = df[df["USA_WIND"].notna()]
 
    numeric_cols = ["USA_WIND", "LAT", "LON", "DIST2LAND", "SEASON"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["USA_WIND"])

    plot_wind_distribution(df)
    plot_feature_importance()

if __name__ == "__main__":
    main()
