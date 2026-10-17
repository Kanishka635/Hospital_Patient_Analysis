
import pandas as pd
import numpy as np
from pathlib import Path

# Locate the project and load the data
project_dir = Path.home() / "Hospital_Patient_Analysis"
file_path = project_dir / "synthetic_hospital_arrivals.csv"

df = pd.read_csv(file_path, parse_dates=["date"])
arrivals = df["arrivals"]

# Use the first 300 days for training
# Use the remaining 65 days for testing
train = arrivals.iloc[:300]
test = arrivals.iloc[300:]

# Seven-day moving-average forecast
history = list(train)
predictions = []

for actual_value in test:
    prediction = np.mean(history[-7:])
    predictions.append(prediction)

    # Use the observed arrival count for the next forecast
    history.append(actual_value)

# Evaluate forecast accuracy
actual = test.to_numpy()
predicted = np.array(predictions)

mae = np.mean(np.abs(actual - predicted))
rmse = np.sqrt(np.mean((actual - predicted) ** 2))

print("=== FORECAST EVALUATION ===")
print("Method: 7-day moving average")
print("Training days:", len(train))
print("Testing days:", len(test))
print(f"MAE: {mae:.2f} patients")
print(f"RMSE: {rmse:.2f} patients")

# Display the first 10 predictions
results = pd.DataFrame({
    "date": df["date"].iloc[300:].dt.strftime("%Y-%m-%d"),
    "actual": actual,
    "predicted": np.round(predicted, 2)
})

print("\n=== FIRST 10 FORECAST RESULTS ===")
print(results.head(10).to_string(index=False))

# Save all test predictions
output_file = project_dir / "forecast_results.csv"
results.to_csv(output_file, index=False)

print("\nForecast results saved to:", output_file)
