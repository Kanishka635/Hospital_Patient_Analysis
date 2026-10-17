
import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

project_dir = Path.home() / "Hospital_Patient_Analysis"
df = pd.read_csv(project_dir / "synthetic_hospital_arrivals.csv")

arrivals = df["arrivals"]
train = arrivals.iloc[:300]
test = arrivals.iloc[300:]

# Method 1: Seven-day moving average
history = list(train)
moving_predictions = []

for actual_value in test:
    moving_predictions.append(np.mean(history[-7:]))
    history.append(actual_value)

# Method 2: Simple exponential smoothing
model = SimpleExpSmoothing(
    train.astype(float),
    initialization_method="estimated"
).fit(optimized=True)

# One-step-ahead predictions using observed test values
# to update the forecast after each day.
smoothing_predictions = []
history = list(train)

for actual_value in test:
    current_model = SimpleExpSmoothing(
        pd.Series(history, dtype=float),
        initialization_method="estimated"
    ).fit(optimized=True)

    smoothing_predictions.append(
        float(current_model.forecast(1).iloc[0])
    )
    history.append(actual_value)

actual = test.to_numpy()
moving_predictions = np.array(moving_predictions)
smoothing_predictions = np.array(smoothing_predictions)

def evaluate(actual, predicted):
    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    return mae, rmse

mae_ma, rmse_ma = evaluate(actual, moving_predictions)
mae_es, rmse_es = evaluate(actual, smoothing_predictions)

comparison = pd.DataFrame({
    "Method": ["7-day Moving Average", "Simple Exponential Smoothing"],
    "MAE": [mae_ma, mae_es],
    "RMSE": [rmse_ma, rmse_es]
}).round(3)

print("\n=== FORECAST METHOD COMPARISON ===")
print(comparison.to_string(index=False))

print("\nLower MAE and RMSE indicate better test accuracy.")
print("Results are based on simulated hospital arrivals.")

comparison.to_csv(
    project_dir / "forecast_method_comparison.csv",
    index=False
)

print("\nSaved: forecast_method_comparison.csv")
