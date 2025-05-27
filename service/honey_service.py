import joblib
import pandas as pd
import os

# Load the trained multi-output model (e.g., Pipeline)
model_path = "honey_model.pkl"
if not os.path.exists(model_path):
    raise FileNotFoundError("Model file not found.")

model = joblib.load(model_path)

# Columns expected by the model
expected_columns = [
    "CS", "Density", "WC", "pH", "EC",
    "F", "G", "Viscosity", "Pollen_analysis"
]

def predict_honey_quality(input_data):
    # Convert to DataFrame
    df = pd.DataFrame(input_data)

    # Check for missing columns
    missing = [col for col in expected_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Predict
    predictions = model.predict(df)

    # Return structured predictions
    results = []
    for i, (purity, price) in enumerate(predictions):
        results.append({
            "sample": i + 1,
            "predicted_purity": round(float(purity), 4),
            "predicted_price": round(float(price), 2)
        })

    return results
