
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy import stats

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(
    page_title="Hospital Patient Arrival Analysis",
    page_icon="🏥",
    layout="wide"
)

PROJECT_DIR = Path.home() / "Hospital_Patient_Analysis"

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    file_path = PROJECT_DIR / "synthetic_hospital_arrivals.csv"
    data = pd.read_csv(file_path)

    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values("date")

    return data

st.title("🏥 Hospital Patient Arrival Analysis")
st.caption("Probability & Statistics Project | Python Analytics Dashboard")

try:
    df = load_data()
except Exception as error:
    st.error(f"Could not load the dataset: {error}")
    st.info("Check that synthetic_hospital_arrivals.csv is in your project folder.")
    st.stop()

if "arrivals" not in df.columns or "date" not in df.columns:
    st.error("The dataset must contain 'date' and 'arrivals' columns.")
    st.stop()

df["day_type"] = np.where(
    df["date"].dt.dayofweek >= 5,
    "Weekend",
    "Weekday"
)

# ---------------- SIDEBAR ----------------
st.sidebar.header("Dashboard Controls")

start_date = st.sidebar.date_input(
    "Start date",
    value=df["date"].min().date(),
    min_value=df["date"].min().date(),
    max_value=df["date"].max().date()
)

end_date = st.sidebar.date_input(
    "End date",
    value=df["date"].max().date(),
    min_value=df["date"].min().date(),
    max_value=df["date"].max().date()
)

if start_date > end_date:
    st.error("Start date must be on or before end date.")
    st.stop()

filtered = df[
    (df["date"].dt.date >= start_date)
    & (df["date"].dt.date <= end_date)
].copy()

if filtered.empty:
    st.warning("No records found for this date range.")
    st.stop()

# ---------------- SUMMARY METRICS ----------------
st.subheader("1. Patient Arrival Summary")

total = int(filtered["arrivals"].sum())
average = filtered["arrivals"].mean()
maximum = int(filtered["arrivals"].max())
minimum = int(filtered["arrivals"].min())

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Patient Arrivals", f"{total:,}")
c2.metric("Average Per Day", f"{average:.2f}")
c3.metric("Highest Daily Arrivals", maximum)
c4.metric("Lowest Daily Arrivals", minimum)

# ---------------- TABS ----------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Daily Trends",
    "Weekday vs Weekend",
    "Probability",
    "Hypothesis Test",
    "Forecasting"
])

# ---------------- TAB 1 ----------------
with tab1:
    st.subheader("Daily Patient Arrivals")

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(
        filtered["date"],
        filtered["arrivals"],
        marker=".",
        linewidth=1
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Patient Arrivals")
    ax.set_title("Daily Hospital Patient Arrivals")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Descriptive Statistics")
    descriptive = filtered["arrivals"].describe().to_frame(
        name="Patient Arrivals"
    )
    st.dataframe(descriptive, use_container_width=True)

    st.download_button(
        "Download filtered data as CSV",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="hospital_arrivals_filtered.csv",
        mime="text/csv"
    )

# ---------------- TAB 2 ----------------
with tab2:
    st.subheader("Weekday and Weekend Comparison")

    group_stats = filtered.groupby("day_type")["arrivals"].agg(
        ["count", "mean", "median", "std", "min", "max"]
    ).round(2)

    st.dataframe(group_stats, use_container_width=True)

    means = filtered.groupby("day_type")["arrivals"].mean()

    fig, ax = plt.subplots(figsize=(7, 4))
    means.reindex(["Weekday", "Weekend"]).dropna().plot(
        kind="bar", ax=ax
    )
    ax.set_xlabel("Day Type")
    ax.set_ylabel("Average Patient Arrivals")
    ax.set_title("Average Arrivals: Weekday vs Weekend")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    if "Weekday" in means.index and "Weekend" in means.index:
        difference = means["Weekday"] - means["Weekend"]
        st.write(
            f"Difference in average daily arrivals: "
            f"{difference:.2f} patients."
        )

# ---------------- TAB 3 ----------------
with tab3:
    st.subheader("Poisson Probability Calculator")

    st.write(
        "This calculator uses a Poisson distribution with "
        "the average arrival rate from the selected date range."
    )

    lam = float(filtered["arrivals"].mean())

    st.metric("Estimated Average Rate (λ)", f"{lam:.2f}")

    event = st.selectbox(
        "Choose a probability",
        [
            "Exactly k patients",
            "More than k patients",
            "Fewer than k patients"
        ]
    )

    k = st.number_input(
        "Enter the patient count k",
        min_value=0,
        max_value=10000,
        value=int(round(lam)),
        step=1
    )

    if event == "Exactly k patients":
        probability = stats.poisson.pmf(int(k), lam)
    elif event == "More than k patients":
        probability = stats.poisson.sf(int(k), lam)
    else:
        probability = stats.poisson.cdf(int(k) - 1, lam)

    st.metric("Calculated Probability", f"{probability:.4%}")

    st.caption(
        "The Poisson model assumes a fixed average rate. "
        "Real arrivals may vary by weekday, season, and other factors."
    )

# ---------------- TAB 4 ----------------
with tab4:
    st.subheader("Welch's Independent Two-Sample t-test")

    weekday_values = filtered.loc[
        filtered["day_type"] == "Weekday", "arrivals"
    ]
    weekend_values = filtered.loc[
        filtered["day_type"] == "Weekend", "arrivals"
    ]

    if len(weekday_values) >= 2 and len(weekend_values) >= 2:
        t_stat, p_value = stats.ttest_ind(
            weekday_values,
            weekend_values,
            equal_var=False
        )

        difference = weekday_values.mean() - weekend_values.mean()

        a, b = st.columns(2)
        a.metric("Difference in Means", f"{difference:.2f}")
        b.metric("p-value", f"{p_value:.6g}")

        st.write(f"t-statistic: {t_stat:.4f}")
        st.write("Significance level: 0.05")

        if p_value < 0.05:
            st.success(
                "Reject the null hypothesis for this dataset: "
                "the weekday and weekend means differ statistically."
            )
        else:
            st.info(
                "Fail to reject the null hypothesis: "
                "there is insufficient evidence of a difference."
            )

        st.markdown(
            "**H₀:** Weekday and weekend mean arrivals are equal.  \n"
            "**H₁:** Weekday and weekend mean arrivals are different."
        )
    else:
        st.warning(
            "Select a date range containing at least two weekdays "
            "and two weekend observations."
        )

# ---------------- TAB 5 ----------------
with tab5:
    st.subheader("Forecasting Method Comparison")

    result_path = PROJECT_DIR / "forecast_method_comparison.csv"

    if result_path.exists():
        results = pd.read_csv(result_path)
        st.dataframe(results, hide_index=True, use_container_width=True)

        best_method = results.loc[results["MAE"].idxmin(), "Method"]
        st.success(f"Best method by MAE on the test set: {best_method}")

        st.write(
            "MAE measures average absolute prediction error. "
            "RMSE penalizes larger errors more strongly. "
            "Lower values indicate better accuracy on the same test data."
        )

        st.download_button(
            "Download forecast comparison",
            data=results.to_csv(index=False).encode("utf-8"),
            file_name="forecast_method_comparison.csv",
            mime="text/csv"
        )
    else:
        st.warning(
            "Forecast comparison file not found. "
            "Run compare_forecasts.py first."
        )

    forecast_path = PROJECT_DIR / "forecast_results.csv"

    if forecast_path.exists():
        forecast = pd.read_csv(forecast_path)
        st.subheader("Actual vs Predicted Arrivals")

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(
            pd.to_datetime(forecast["date"]),
            forecast["actual"],
            label="Actual",
            linewidth=1.5
        )
        ax.plot(
            pd.to_datetime(forecast["date"]),
            forecast["predicted"],
            label="7-day Moving Average",
            linewidth=1.5
        )
        ax.set_xlabel("Date")
        ax.set_ylabel("Patient Arrivals")
        ax.set_title("Forecast Evaluation: Test Period")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

# ---------------- FOOTER ----------------
st.divider()
st.caption(
    "Academic demonstration only. The current dataset is simulated, "
    "not real hospital data. Forecasts are not clinical decisions."
)
