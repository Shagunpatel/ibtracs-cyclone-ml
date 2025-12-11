# test.py
import joblib
from sklearn.metrics import mean_absolute_error, r2_score
from train import load_and_preprocess, MODEL_PATH

def test():
    # Reload data and same split
    X_train, X_test, y_train, y_test = load_and_preprocess()

    # Load trained model
    print(f"Loading model from {MODEL_PATH} ...")
    model = joblib.load(MODEL_PATH)

    # Evaluate
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print("=== TEST RESULTS ===")
    print(f"Test MAE: {mae:.2f}")
    print(f"Test R^2:  {r2:.3f}")

if __name__ == "__main__":
    test()
