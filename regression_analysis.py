
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Project folder and dataset
project_dir = Path(__file__).resolve().parent
data_path = project_dir / "synthetic_hospital_arrivals.csv"

df = pd.read_csv(data_path)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# Input: day number
# Target: daily patient arrivals
X = np.arange(len(df), dtype=float)
y = df["arrivals"].to_numpy(dtype=float)

# Split chronologically: first 80% for training, last 20% for testing
split = int(len(df) * 0.80)

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Fit a straight-line regression using training data only
model = linregress(X_train, y_train)

intercept = model.intercept
slope = model.slope

# Predict arrivals for the test period
y_pred = intercept + slope * X_test

# Evaluate predictions
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# Save actual and predicted results
results = pd.DataFrame({
    "date": df["date"].iloc[split:].to_numpy(),
    "actual_arrivals": y_test,
    "predicted_arrivals": y_pred
})

results.to_csv(project_dir / "regression_results.csv", index=False)

# Display results
print("\n=== LINEAR REGRESSION ANALYSIS ===")
print(f"Total observations: {len(df)}")
print(f"Training observations: {len(X_train)}")
print(f"Testing observations: {len(X_test)}")
print(f"Regression equation: predicted arrivals = {intercept:.3f} + ({slope:.4f} * day number)")
print(f"MAE: {mae:.3f} patients")
print(f"RMSE: {rmse:.3f} patients")
print(f"Test R-squared: {r2:.3f}")

# Plot actual and predicted arrivals
plt.figure(figsize=(12, 5))
plt.plot(df["date"].iloc[split:], y_test, label="Actual arrivals", marker=".")
plt.plot(df["date"].iloc[split:], y_pred, label="Regression predictions")
plt.xlabel("Date")
plt.ylabel("Patient arrivals")
plt.title("Linear Regression: Actual vs Predicted Arrivals")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(project_dir / "regression_test_results.png", dpi=150)
plt.show()

print("\nSaved: regression_results.csv")
print("Saved: regression_test_results.png")
print("\nNote: The dataset is simulated and the regression uses day number only.")
