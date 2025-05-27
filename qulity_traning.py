import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# 1. Load the CSV dataset
#    Replace 'Book1.csv' with your actual CSV filename or path
df = pd.read_csv("honey_purity_dataset.csv")

# 2. Separate features (X) and targets (y)
#    We'll predict both Purity and Price in a multi-output setting
X = df.drop(["Purity", "Price"], axis=1)
y = df[["Purity", "Price"]]

# 3. Identify numeric and categorical columns
numeric_features = ["CS", "Density", "WC", "pH", "EC", "F", "G", "Viscosity"]
categorical_features = ["Pollen_analysis"]

# 4. Create preprocessing pipelines
#    - Scale numeric features
#    - One-hot encode categorical features
numeric_transformer = Pipeline([
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("onehot", OneHotEncoder(drop="first"))  # drop="first" avoids the dummy variable trap
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# 5. Build a pipeline:
#    - Preprocessing
#    - RandomForestRegressor with parameters to show real-time progress (verbose=2)
#      and possibly improved performance (e.g., n_estimators=200).
model = Pipeline([
    ("preprocessor", preprocessor),
    ("rf", RandomForestRegressor(
        n_estimators=50,
        max_depth=20,       # Adjust as needed for your dataset
        random_state=42,
        verbose=2,          # <-- Real-time training progress
        n_jobs=-1           # Use all available CPU cores
    ))
])

# 6. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7. Fit (train) the model
print("=== Training Started ===")
model.fit(X_train, y_train)
print("=== Training Finished ===\n")

# 8. Predict on the test set
predictions = model.predict(X_test)

# 9. Evaluate the model for both Purity and Price
mse_purity = mean_squared_error(y_test["Purity"], predictions[:, 0])
mse_price = mean_squared_error(y_test["Price"], predictions[:, 1])
r2_purity = r2_score(y_test["Purity"], predictions[:, 0])
r2_price = r2_score(y_test["Price"], predictions[:, 1])

print("=== Evaluation Results ===")
print(f"Purity MSE: {mse_purity:.4f} | Purity R²: {r2_purity:.4f}")
print(f"Price  MSE: {mse_price:.4f}  | Price  R²: {r2_price:.4f}")

# (Optional) Show a quick histogram of residuals for Purity
purity_residuals = y_test["Purity"] - predictions[:, 0]
plt.hist(purity_residuals, bins=20, edgecolor='k')
plt.title("Purity Residuals Distribution")
plt.xlabel("Residual (Actual - Predicted)")
plt.ylabel("Frequency")
plt.show()

# 10. Save the trained model
joblib.dump(model, "honey_model.pkl")
print("\nModel saved as 'honey_model.pkl'")
