# train.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# Paths
DATA_PATH = "data/ibtracs.last3years.list.v04r01.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "random_forest_usa_wind.pkl")

def load_and_preprocess(data_path: str = DATA_PATH):
    """
    Load the IBTrACS CSV and prepare features/target for modeling.

    Target: USA_WIND (storm wind speed in knots)
    Features: LAT, LON, DIST2LAND, SEASON

    This function:
    - selects those columns
    - coerces them to numeric (invalid strings -> NaN)
    - drops rows with NaNs
    - returns a train/test split
    """
    print(f"Loading data from {data_path} ...")
    df = pd.read_csv(data_path, low_memory=False)

    # Keep only needed columns
    cols = ["USA_WIND", "LAT", "LON", "DIST2LAND", "SEASON"]
    df = df[cols]

    # Coerce to numeric to get rid of strings like 'degrees_north'
    numeric_cols = cols  # all of them should be numeric for our purpose
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop rows with missing values AFTER conversion
    df = df.dropna()
    print(f"Data shape after coercing to numeric and dropping NaNs: {df.shape}")

    # Define features (X) and target (y)
    feature_cols = ["LAT", "LON", "DIST2LAND", "SEASON"]
    X = df[feature_cols]
    y = df["USA_WIND"]

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Train size: {X_train.shape[0]}  |  Test size: {X_test.shape[0]}")
    print("Feature dtypes:")
    print(X_train.dtypes)

    return X_train, X_test, y_train, y_test

def train():
    # Ensure model directory exists
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Load and preprocess data
    X_train, X_test, y_train, y_test = load_and_preprocess()

    # Define model
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    # Train model
    print("Training RandomForestRegressor ...")
    model.fit(X_train, y_train)

    # Evaluate on test split inside training script for quick feedback
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"Training complete.")
    print(f"Test MAE: {mae:.2f}")
    print(f"Test R^2:  {r2:.3f}")

    # Save the trained model
    joblib.dump(model, MODEL_PATH)
    print(f"Saved trained model to {MODEL_PATH}")

if __name__ == "__main__":
    train()
