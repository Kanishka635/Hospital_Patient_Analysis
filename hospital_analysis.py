
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import poisson, ttest_ind
from pathlib import Path

# Project folder and dataset
PROJECT_DIR = Path.home() / "Hospital_Patient_Analysis"
DATA_FILE = PROJECT_DIR / "synthetic_hospital_arrivals.csv"

df = pd.read_csv(DATA_FILE)
df["date"] = pd.to_datetime(df["date"])

# 1. Descriptive statistics
print("\n=== DESCRIPTIVE STATISTICS ===")
print(df["arrivals"].describe().round(2))

# 2. Weekday and weekend comparison
df["day_type"] = df["date"].dt.dayofweek.map(
    lambda d: "Weekend" if d >= 5 else "Weekday"
)

print("\n=== WEEKDAY VS WEEKEND ===")
print(
    df.groupby("day_type")["arrivals"]
    .agg(["count", "mean", "median", "std", "min", "max"])
    .round(2)
)

# 3. Poisson probabilities
rate = df["arrivals"].mean()

print("\n=== POISSON PROBABILITIES ===")
print(f"Average arrival rate: {rate:.2f}")
print(f"Exactly 45 arrivals: {poisson.pmf(45, rate)*100:.2f}%")
print(f"More than 50 arrivals: {(1-poisson.cdf(50, rate))*100:.2f}%")
print(f"Fewer than 40 arrivals: {poisson.cdf(39, rate)*100:.2f}%")

# 4. Welch's t-test
weekday = df.loc[df["day_type"] == "Weekday", "arrivals"]
weekend = df.loc[df["day_type"] == "Weekend", "arrivals"]

t_stat, p_value = ttest_ind(weekday, weekend, equal_var=False)

print("\n=== HYPOTHESIS TEST ===")
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.10g}")
print(
    "Decision:",
    "Reject the null hypothesis"
    if p_value < 0.05
    else "Fail to reject the null hypothesis"
)

# 5. Save weekday/weekend comparison graph
means = df.groupby("day_type")["arrivals"].mean().reindex(
    ["Weekday", "Weekend"]
)

ax = means.plot(
    kind="bar",
    figsize=(7, 5),
    title="Average Daily Hospital Arrivals: Weekday vs Weekend",
    xlabel="Day Type",
    ylabel="Average Patients per Day",
    rot=0,
)

ax.bar_label(ax.containers[0], fmt="%.2f")
plt.tight_layout()
plt.savefig(PROJECT_DIR / "weekday_weekend_comparison.png", dpi=300)
plt.close()

print("\nGraph saved in:", PROJECT_DIR)
print("Analysis completed successfully.")
print("Note: The dataset is simulated, not real hospital data.")
