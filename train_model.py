import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
import joblib
import json
import os

# -------- CONFIG --------
INPUT_FILE = "iot_anomaly_dataset.json"  # Change if needed
MODEL_PATH = "phase3_predict_model.pkl"
SCALER_PATH = "phase3_predict_scaler.pkl"
# ------------------------

def load_data(file_path):
    """Load IoT dataset from CSV, JSON array, or NDJSON."""
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    elif file_path.endswith(".json"):
        with open(file_path, "r") as f:
            try:
                # Try full JSON load
                df = pd.DataFrame(json.load(f))
            except json.JSONDecodeError:
                # Fallback: NDJSON parsing
                f.seek(0)
                records = []
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass  # Skip malformed lines
                df = pd.DataFrame(records)
    else:
        raise ValueError("Unsupported file format. Use CSV or JSON.")
    return df

def prepare_features(df):
    """Add rolling mean/std features."""
    w = 3
    for c in ["voltage_V", "energy_usage_J"]:
        if c in df.columns:
            df[f"{c}_roll_mean"] = df[c].rolling(window=w, min_periods=1).mean()
            df[f"{c}_roll_std"] = df[c].rolling(window=w, min_periods=1).std().fillna(0.0)
    feature_cols = [
        "voltage_V", "energy_usage_J", "temperature_C", "humidity_%", "field_strength_V_m",
        "storage_level_%", "sensor_active",
        "voltage_V_roll_mean", "voltage_V_roll_std",
        "energy_usage_J_roll_mean", "energy_usage_J_roll_std"
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]
    return df, feature_cols

def main():
    print(f"[INFO] Loading dataset from {INPUT_FILE}")
    df = load_data(INPUT_FILE)
    print(f"[INFO] Loaded {len(df)} rows.")

    print("[INFO] Preparing features...")
    df, feature_cols = prepare_features(df)

    if not feature_cols:
        print(f"[ERROR] No expected feature columns found in dataset.")
        print(f"[INFO] Dataset columns: {list(df.columns)}")
        print(f"[HINT] Ensure your file has at least these columns: "
            f"voltage_V, energy_usage_J, temperature_C, humidity_%, "
            f"field_strength_V_m, storage_level_%, sensor_active")
        return

    print("[INFO] Scaling features...")
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(df[feature_cols])

    print("[INFO] Training Isolation Forest...")
    model = IsolationForest(
        contamination=0.05,
        random_state=42,
        n_estimators=100
    )
    model.fit(X_scaled)

    print(f"[INFO] Saving model to {MODEL_PATH}")
    joblib.dump(model, MODEL_PATH)

    print(f"[INFO] Saving scaler to {SCALER_PATH}")
    joblib.dump(scaler, SCALER_PATH)

    # -------- Accuracy Check --------
    if "pred_is_anomaly" in df.columns or "actual_anomaly" in df.columns:
        print("[INFO] Running accuracy check...")
        preds = model.predict(X_scaled)
        preds = (preds == -1).astype(int)  # Convert to 1 = anomaly
        true_labels = df["pred_is_anomaly"] if "pred_is_anomaly" in df.columns else df["actual_anomaly"]
        print(classification_report(true_labels, preds))
    else:
        print("[INFO] No ground truth labels found. Skipping accuracy check.")

    print("[SUCCESS] Model and scaler saved successfully.")

if __name__ == "__main__":
    main()
