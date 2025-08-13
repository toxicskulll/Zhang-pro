import pandas as pd
import joblib
import sys
import json

MODEL_PATH = "phase3_predict_model.pkl"
SCALER_PATH = "phase3_predict_scaler.pkl"

def load_model():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

def prepare_features(df):
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

def apply_decision_logic(df):
    decisions = []
    for _, row in df.iterrows():
        if row["pred_is_anomaly"] == 1:
            decisions.append(("diagnostic", "low_power"))
        else:
            storage = row.get("storage_level_%", 50)
            field_strength = row.get("field_strength_V_m", 200)
            if storage > 70 and field_strength > 300:
                decisions.append(("max", "full"))
            elif storage < 30 and field_strength < 200:
                decisions.append(("normal", "low_power"))
            else:
                decisions.append(("normal", "normal"))
    df["harvesting_mode"] = [h for h, _ in decisions]
    df["sensor_mode"] = [s for _, s in decisions]
    return df

def predict(input_path, output_path=None):
    model, scaler = load_model()
    if input_path.endswith(".csv"):
        df = pd.read_csv(input_path)
    elif input_path.endswith(".json"):
        with open(input_path, "r") as f:
            df = pd.DataFrame(json.load(f))
    else:
        raise ValueError("Unsupported file format. Use CSV or JSON.")
    df, feature_cols = prepare_features(df)
    X_scaled = scaler.transform(df[feature_cols])
    preds = model.predict(X_scaled)
    scores = -model.score_samples(X_scaled)
    df["pred_is_anomaly"] = (preds == -1).astype(int)
    df["anomaly_score"] = scores
    df = apply_decision_logic(df)
    if output_path:
        if output_path.endswith(".csv"):
            df.to_csv(output_path, index=False)
        elif output_path.endswith(".json"):
            df.to_json(output_path, orient="records", date_format="iso")
        else:
            raise ValueError("Output must end with .csv or .json")
    else:
        print(df[["timestamp", "pred_is_anomaly", "anomaly_score", "harvesting_mode", "sensor_mode"]])

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <input_file> [<output_file>]")
    else:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        predict(input_file, output_file)
