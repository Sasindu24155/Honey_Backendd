import joblib
import pandas as pd

# 1. Load the trained model (pipeline)
model = joblib.load("honey_model.pkl")

# 2. Create a new sample or multiple samples as a DataFrame
#    Make sure the columns match exactly what the model expects (same names and order).
new_data = {
    "CS": [5.5, 7.2],
    "Density": [1.75, 1.80],
    "WC": [13.0, 14.5],
    "pH": [4.2, 3.9],
    "EC": [0.45, 0.52],
    "F": [40, 45],
    "G": [35, 30],
    "Viscosity": [7000, 8000],
    "Pollen_analysis": ["Manuka", "Clover"]  # Example floral sources
}

# Convert the dictionary to a DataFrame
X_new = pd.DataFrame(new_data)

# 3. Predict using the model
predictions = model.predict(X_new)

# 4. Parse out Purity (index 0) and Price (index 1) from the multi-output array
#    'predictions[i, 0]' is Purity; 'predictions[i, 1]' is Price for the i-th sample
for i, (purity, price) in enumerate(predictions):
    print(f"=== Sample {i+1} ===")
    print(f"Predicted Purity: {purity:.4f}")
    print(f"Predicted Price:  {price:.2f}\n")
