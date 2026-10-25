from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(
    page_title="Hospital Patient Arrival Analysis",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_DIR = Path(__file__).resolve().parent
DATA_FILE = PROJECT_DIR / "synthetic_hospital_arrivals.csv"

# ---------------- PROFESSIONAL BLUE-AND-WHITE THEME ----------------
BLUE = "#145DA0"
DARK_BLUE = "#0B2E4F"
PALE_BLUE = "#EAF3FB"
PAGE_BG = "#F5F8FC"
TEXT = "#18324D"
MUTED = "#64788C"
BORDER = "#D9E4EF"

st.markdown(
    f"""
    <style>
    :root {{
        --primary-color: {BLUE};
        --background-color: {PAGE_BG};
        --secondary-background-color: #FFFFFF;
        --text-color: {TEXT};
        color-scheme: light;
    }}
    .stApp {{
        background: {PAGE_BG};
        color: {TEXT};
    }}
    [data-testid="stHeader"] {{
        background: rgba(245, 248, 252, 0.95);
    }}
       /* Sidebar background */
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] > div {{
        background-color: #EAF3FB !important;
        color: #0B2E4F !important;
    }}

    /* Navigation labels */
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] [role="radiogroup"] * {{
        color: #0B2E4F !important;
    }}

    /* Date input fields: white background, dark text */
    [data-testid="stSidebar"] [data-testid="stDateInput"] input {{
        background-color: #FFFFFF !important;
        color: #17324D !important;
        -webkit-text-fill-color: #17324D !important;
        border: 1px solid #B8C9DA !important;
        border-radius: 6px !important;
    }}
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {{
        color: {DARK_BLUE};
    }}
    h1, h2, h3 {{
        color: {DARK_BLUE} !important;
        letter-spacing: -0.02em;
    }}
    [data-testid="stMetric"] {{
        background: #FFFFFF;
        border: 1px solid {BORDER};
        border-top: 4px solid {BLUE};
        border-radius: 12px;
        padding: 16px 18px;
        box-shadow: 0 2px 8px rgba(11, 46, 79, 0.04);
    }}
    [data-testid="stMetricLabel"] {{
        color: {MUTED};
        font-weight: 600;
    }}
    [data-testid="stMetricValue"] {{
        color: {DARK_BLUE};
    }}
    div.stButton > button, div.stDownloadButton > button {{
        background: {BLUE};
        color: white;
        border: 1px solid {BLUE};
        border-radius: 8px;
        font-weight: 600;
    }}
    div.stButton > button:hover, div.stDownloadButton > button:hover {{
        background: {DARK_BLUE};
        color: white;
        border-color: {DARK_BLUE};
    }}
    [data-testid="stAlert"] {{
        border-radius: 10px;
    }}
    div[data-baseweb="input"] > div,
    div[data-baseweb="select"] > div {{
        border-color: {BORDER};
        border-radius: 8px;
    }}
    hr {{
        border-color: {BORDER};
    }}
    .hero {{
        background: linear-gradient(110deg, #0B2E4F 0%, #145DA0 70%, #2C82C9 100%);
        padding: 24px 28px;
        border-radius: 16px;
        margin-bottom: 18px;
        color: white;
    }}
    .hero h1 {{
        color: white !important;
        margin: 0 0 6px 0;
        font-size: 2rem;
    }}
    .hero p {{
        color: #EAF3FB;
        margin: 0;
        font-size: 1rem;
    }}
    .notice {{
        background: #FFF8E7;
        color: #684D10;
        border: 1px solid #F0D99A;
        border-left: 5px solid #D6A328;
        border-radius: 9px;
        padding: 12px 15px;
        margin: 12px 0 20px 0;
    }}
    .section-note {{
        color: {MUTED};
        margin-top: -8px;
        margin-bottom: 14px;
    }}

    
    /* Keep the sidebar toggle arrow visible */
    [data-testid="stHeader"] button,
    [data-testid="stSidebarCollapsedControl"] button {{
        color: #0B2E4F !important;
        background-color: #EAF3FB !important;
        opacity: 1 !important;
        visibility: visible !important;
    }}

    [data-testid="stHeader"] button svg,
    [data-testid="stSidebarCollapsedControl"] button svg {{
        color: #0B2E4F !important;
        fill: #0B2E4F !important;
        stroke: #0B2E4F !important;
        opacity: 1 !important;
        visibility: visible !important;
    }}


    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- DATA LOADING ----------------
@st.cache_data
def load_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_FILE}. Put "
            "'synthetic_hospital_arrivals.csv' beside app.py."
        )
    data = pd.read_csv(DATA_FILE)
    if "date" not in data.columns or "arrivals" not in data.columns:
        raise ValueError("CSV must contain 'date' and 'arrivals' columns.")
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["arrivals"] = pd.to_numeric(data["arrivals"], errors="coerce")
    data = data.dropna(subset=["date", "arrivals"]).sort_values("date")
    if data.empty:
        raise ValueError("The dataset has no valid date/arrival rows.")
    data["day_type"] = np.where(data["date"].dt.dayofweek >= 5, "Weekend", "Weekday")
    return data

try:
    df = load_data()
except Exception as exc:
    st.error(f"Could not load the dataset: {exc}")
    st.stop()

# ---------------- SIDEBAR: NAVIGATION AND FILTERS ----------------
with st.sidebar:
    st.markdown("## 🏥 Hospital Analytics")
    st.caption("Probability & Statistics Project")
    st.divider()
    page = st.radio(
        "NAVIGATION",
        [
            "Overview & Daily Trends",
            "Weekday vs Weekend",
            "Probability",
            "Hypothesis Test",
            "Regression Analysis",
            "Distribution & Skewness",
            "Normal Distribution",
            "Confidence Interval",
            "Sampling & CLT",
            "ANOVA Test",
            "Least-Squares Curve Fitting",
	    "Live Patient Arrivals",
            
            
        ],
        label_visibility="visible",
    )
    st.divider()
    st.markdown("### Date filters")
    min_date = df["date"].min().date()
    max_date = df["date"].max().date()
    start_date = st.date_input(
        "Start date",
        value=min_date,
        min_value=min_date,
        max_value=max_date,
        key="start_date",
    )
    end_date = st.date_input(
        "End date",
        value=max_date,
        min_value=min_date,
        max_value=max_date,
        key="end_date",
    )
    st.caption(f"Available data: {min_date:%d %b %Y} – {max_date:%d %b %Y}")

if start_date > end_date:
    st.error("Start date must be on or before the end date.")
    st.stop()

filtered = df[
    (df["date"].dt.date >= start_date) & (df["date"].dt.date <= end_date)
].copy()

if filtered.empty:
    st.warning("No records are available for this date range. Choose another range in the sidebar.")
    st.stop()

# ---------------- HEADER AND SIMULATED-DATA NOTICE ----------------
st.markdown(
    """
    <div class="hero">
      <h1>Hospital Patient Arrival Analysis</h1>
      <p>Explore patient-arrival patterns using statistical summaries, probability,
      hypothesis testing, and linear regression.</p>
    </div>

    """,
    unsafe_allow_html=True,
)

total = int(filtered["arrivals"].sum())
average = float(filtered["arrivals"].mean())
maximum = int(filtered["arrivals"].max())
minimum = int(filtered["arrivals"].min())

# ---------------- PAGE: OVERVIEW AND DAILY TRENDS ----------------
if page == "Overview & Daily Trends":
    st.subheader("Patient arrival summary")
    st.markdown(
        f'<p class="section-note">Selected period: {start_date:%d %b %Y} – {end_date:%d %b %Y}</p>',
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total arrivals", f"{total:,}")
    c2.metric("Average per day", f"{average:.2f}")
    c3.metric("Highest daily count", f"{maximum:,}")
    c4.metric("Lowest daily count", f"{minimum:,}")

    st.divider()
    st.subheader("Daily patient arrivals")
    fig, ax = plt.subplots(figsize=(11, 4.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.plot(filtered["date"], filtered["arrivals"], color=BLUE, linewidth=2, marker="o", markersize=3)
    ax.fill_between(filtered["date"], filtered["arrivals"], color="#8DBCE8", alpha=0.22)
    ax.set_xlabel("Date")
    ax.set_ylabel("Patient arrivals")
    ax.grid(True, axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


    # ---------------- DESCRIPTIVE STATISTICS ----------------
    st.subheader("Descriptive Statistics")
    st.caption("A simple summary of patient arrivals during the selected dates.")

    arrivals = filtered["arrivals"]

    descriptive = pd.DataFrame({
        "Statistic": [
            "Number of days",
            "Average (mean)",
            "Middle value (median)",
            "Most common value (mode)",
            "Standard deviation",
            "Variance",
            "Lowest daily arrivals",
            "25th percentile (Q1)",
            "75th percentile (Q3)",
            "Highest daily arrivals",
        ],
        "Value": [
            f"{arrivals.count():,.0f}",
            f"{arrivals.mean():.2f}",
            f"{arrivals.median():.2f}",
            f"{arrivals.mode().iloc[0]:.2f}",
            f"{arrivals.std():.2f}",
            f"{arrivals.var():.2f}",
            f"{arrivals.min():,.0f}",
            f"{arrivals.quantile(0.25):.2f}",
            f"{arrivals.quantile(0.75):.2f}",
            f"{arrivals.max():,.0f}",
        ],
        "What it tells us": [
            "Number of daily records analysed",
            "Average patients arriving per day",
            "Middle value of daily arrivals",
            "Most frequently occurring daily count",
            "Typical spread around the average, in patients",
            "Measure of variation in daily counts, in squared units",
            "Smallest daily patient count",
            "25% of days have this many arrivals or fewer",
            "75% of days have this many arrivals or fewer",
            "Largest daily patient count",
        ],
    })

    st.dataframe(
        descriptive,
        hide_index=True,
        use_container_width=True,
        height=390,
        column_config={
            "Statistic": st.column_config.TextColumn(
                "Statistic", width="medium"
            ),
            "Value": st.column_config.TextColumn(
                "Result", width="small"
            ),
            "What it tells us": st.column_config.TextColumn(
                "Interpretation", width="large"
            ),
        },
    )

    st.divider()

    st.subheader("Daily Records Overview")
    st.caption("Quick facts for the selected date range.")

    st.metric(
        "Total patient arrivals",
        f"{total:,}",
        help="Total arrivals across all selected days."
    )

    st.metric(
        "Average arrivals per day",
        f"{average:.2f}",
        help="Total arrivals divided by the number of selected days."
    )

    st.metric(
        "Median daily arrivals",
        f"{filtered['arrivals'].median():.2f}",
        help="The middle daily arrival count."
    )


    # ---------------- PAGE: WEEKDAY VS WEEKEND ----------------

elif page == "Weekday vs Weekend":
    st.subheader("Weekday and weekend comparison")
    group_stats = filtered.groupby("day_type")["arrivals"].agg(
        Days="count", Mean="mean", Median="median", Std_Dev="std", Minimum="min", Maximum="max"
    ).round(2)
    st.dataframe(group_stats, use_container_width=True)

    means = filtered.groupby("day_type")["arrivals"].mean().reindex(["Weekday", "Weekend"]).dropna()
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    bars = ax.bar(means.index, means.values, color=[BLUE, "#79A9D8"], width=0.55)
    ax.bar_label(bars, fmt="%.2f", padding=4)
    ax.set_ylabel("Average daily arrivals")
    ax.grid(True, axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    if "Weekday" in means.index and "Weekend" in means.index:
        difference = float(means["Weekday"] - means["Weekend"])
        a, b = st.columns(2)
        a.metric("Difference in means (weekday − weekend)", f"{difference:.2f}")
        b.metric("Weekday / weekend mean ratio", f"{means['Weekday'] / means['Weekend']:.2f}×" if means["Weekend"] else "N/A")


# ---------------- PAGE: PROBABILITY ----------------
elif page == "Probability":
    st.subheader("Poisson probability calculator")
    st.write("Estimate the probability of a daily patient count using the average arrival rate in the selected date range.")
    lam = float(filtered["arrivals"].mean())
    a, b = st.columns(2)
    a.metric("Estimated average rate (λ)", f"{lam:.2f}")
    event = st.selectbox(
        "Probability to calculate",
        ["Exactly k patients", "More than k patients", "Fewer than k patients"],
    )
    k = st.number_input(
        "Patient count k",
        min_value=0,
        max_value=10000,
        value=max(0, int(round(lam))),
        step=1,
    )
    if event == "Exactly k patients":
        probability = stats.poisson.pmf(int(k), lam)
    elif event == "More than k patients":
        probability = stats.poisson.sf(int(k), lam)
    else:
        probability = stats.poisson.cdf(int(k) - 1, lam)
    b.metric("Calculated probability", f"{probability:.4%}")


# ---------------- PAGE: HYPOTHESIS TEST ----------------
elif page == "Hypothesis Test":
    st.subheader("Welch’s independent two-sample t-test")
    weekday_values = filtered.loc[filtered["day_type"] == "Weekday", "arrivals"]
    weekend_values = filtered.loc[filtered["day_type"] == "Weekend", "arrivals"]

    if len(weekday_values) >= 2 and len(weekend_values) >= 2:
        t_stat, p_value = stats.ttest_ind(weekday_values, weekend_values, equal_var=False)
        difference = float(weekday_values.mean() - weekend_values.mean())
        a, b, c = st.columns(3)
        a.metric("Difference in means", f"{difference:.2f}")
        b.metric("t-statistic", f"{t_stat:.4f}")
        c.metric("p-value", f"{p_value:.6g}")
        st.write("**Significance level (α):** 0.05")
        st.markdown("**H₀:** Weekday and weekend mean arrivals are equal.  \n**H₁:** Weekday and weekend mean arrivals are different.")
        if p_value < 0.05:
            st.success("Reject H₀ for this simulated dataset: the weekday and weekend means differ statistically.")
        else:
            st.info("Fail to reject H₀: this selected period does not provide sufficient evidence of a difference.")
        st.caption("Because the dataset is simulated, this test demonstrates the method; it does not establish a real-world hospital effect.")
    else:
        st.warning("Select a date range containing at least two weekdays and two weekend observations.")

# ---------------- PAGE: REGRESSION ANALYSIS ----------------
elif page == "Regression Analysis":
    st.subheader("Linear regression: actual vs predicted arrivals")
    st.markdown(
        '<p class="section-note">A simple linear model uses the position of each day in the selected period to estimate its arrival count.</p>',
        unsafe_allow_html=True,
    )
    reg_df = filtered[["date", "arrivals"]].dropna().copy()
    if len(reg_df) >= 3 and reg_df["arrivals"].nunique() > 1:
        x = np.arange(len(reg_df)).reshape(-1, 1)
        y = reg_df["arrivals"].to_numpy(dtype=float)
        model = LinearRegression()
        model.fit(x, y)
        predictions = model.predict(x)
        mae = mean_absolute_error(y, predictions)
        rmse = float(np.sqrt(mean_squared_error(y, predictions)))
        r2 = r2_score(y, predictions)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mean absolute error (MAE)", f"{mae:.2f}")
        c2.metric("Root mean squared error (RMSE)", f"{rmse:.2f}")
        c3.metric("R² score", f"{r2:.3f}")
        c4.metric("Daily trend slope", f"{model.coef_[0]:+.3f}")

        fig, ax = plt.subplots(figsize=(11, 4.5))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        ax.scatter(reg_df["date"], y, color="#79A9D8", s=26, alpha=0.8, label="Actual arrivals")
        ax.plot(reg_df["date"], predictions, color=BLUE, linewidth=2.5, label="Linear regression fit")
        ax.set_xlabel("Date")
        ax.set_ylabel("Patient arrivals")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.subheader("Model evaluation")
        eval_left, eval_right = st.columns(2)
        with eval_left:
            st.markdown("**What the metrics mean**")
            st.write("- **MAE:** average absolute gap between actual and fitted counts; lower is better.")
            st.write("- **RMSE:** penalizes larger errors more strongly; lower is better.")
            st.write("- **R²:** proportion of variation explained by the fitted line; closer to 1 indicates a better in-sample fit.")
        with eval_right:
            st.markdown("**Model details**")
            st.write(f"- Observations evaluated: **{len(reg_df)} days**")
            st.write(f"- Intercept: **{model.intercept_:.3f}**")
            st.write(f"- Slope: **{model.coef_[0]:.3f} arrivals/day index**")
            st.write("Metrics are calculated on the same selected observations used to fit the line, so they are in-sample evaluation metrics.")
        results = reg_df.copy()
        results["predicted_arrivals"] = predictions.round(3)
        results["residual"] = (y - predictions).round(3)
        with st.expander("View actual, predicted, and residual values"):
            st.dataframe(results, use_container_width=True)
    
    else:
        st.warning("Choose a date range with at least three observations and some variation in arrival counts.")

# ---------------- PAGE: DISTRIBUTION AND SKEWNESS ----------------
elif page == "Distribution & Skewness":
    st.subheader("Distribution, spread, and skewness")
    arrivals = filtered["arrivals"].dropna()
    if len(arrivals) >= 3:
        left, right = st.columns(2)
        with left:
            st.markdown("#### Histogram")
            fig, ax = plt.subplots(figsize=(7, 4))
            fig.patch.set_facecolor("white")
            ax.set_facecolor("white")
            ax.hist(arrivals, bins="auto", color="#79A9D8", edgecolor="white", alpha=0.95)
            ax.set_xlabel("Daily patient arrivals")
            ax.set_ylabel("Number of days")
            ax.grid(True, axis="y", alpha=0.22)
            ax.spines[["top", "right"]].set_visible(False)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        with right:
            st.markdown("#### Box plot")
            fig, ax = plt.subplots(figsize=(7, 4))
            fig.patch.set_facecolor("white")
            ax.set_facecolor("white")
            ax.boxplot(arrivals, vert=False, patch_artist=True, boxprops={"facecolor": "#C7DFF5", "color": BLUE}, medianprops={"color": DARK_BLUE, "linewidth": 2}, whiskerprops={"color": BLUE}, capprops={"color": BLUE}, flierprops={"markerfacecolor": "#79A9D8", "markeredgecolor": BLUE})
            ax.set_xlabel("Daily patient arrivals")
            ax.grid(True, axis="x", alpha=0.22)
            ax.spines[["top", "right"]].set_visible(False)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        q1, median, q3 = arrivals.quantile(0.25), arrivals.median(), arrivals.quantile(0.75)
        skewness = arrivals.skew()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("First quartile (Q1)", f"{q1:.2f}")
        c2.metric("Median", f"{median:.2f}")
        c3.metric("Third quartile (Q3)", f"{q3:.2f}")
        c4.metric("Skewness", f"{skewness:.4f}")
        st.write(f"**Interquartile range (IQR):** {q3 - q1:.2f}")
        if pd.isna(skewness):
            st.info("Skewness cannot be interpreted for constant or unsuitable data.")
        elif skewness > 0.5:
            st.info("Right-skewed: the distribution has a longer tail toward higher arrival counts.")
        elif skewness < -0.5:
            st.info("Left-skewed: the distribution has a longer tail toward lower arrival counts.")
        else:
            st.info("The box covers the middle 50% of observations. Potential outliers are not automatically data errors.")
    else:
        st.warning("Select a date range containing at least three valid daily observations.")


# ---------------- PAGE: NORMAL DISTRIBUTION ----------------
elif page == "Normal Distribution":
    st.subheader("Normal Distribution Analysis")
    st.write(
        "Model daily patient arrivals using a normal distribution "
        "based on the selected date range."
    )

    arrivals = filtered["arrivals"].dropna()
    mean_arrivals = float(arrivals.mean())
    std_arrivals = float(arrivals.std(ddof=1)) if len(arrivals) >= 2 else 0.0

    if len(arrivals) < 2 or std_arrivals == 0:
        st.warning(
            "Select at least two observations with some variation "
            "in daily arrivals to calculate the normal distribution."
        )
    else:
        st.markdown("### Distribution summary")

        c1, c2, c3 = st.columns(3)
        c1.metric("Mean (μ)", f"{mean_arrivals:.2f} patients/day")
        c2.metric("Standard deviation (σ)", f"{std_arrivals:.2f}")
        c3.metric("Observations", f"{len(arrivals)} days")

        st.divider()
        st.markdown("### Normal distribution curve")

        x = np.linspace(
            mean_arrivals - 4 * std_arrivals,
            mean_arrivals + 4 * std_arrivals,
            500,
        )
        density = stats.norm.pdf(x, loc=mean_arrivals, scale=std_arrivals)

        fig, ax = plt.subplots(figsize=(11, 4.5))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        ax.plot(x, density, color=BLUE, linewidth=2.5)
        ax.fill_between(x, density, color="#8DBCE8", alpha=0.30)
        ax.axvline(
            mean_arrivals,
            color=DARK_BLUE,
            linestyle="--",
            linewidth=2,
            label=f"Mean = {mean_arrivals:.2f}",
        )
        ax.set_xlabel("Daily patient arrivals")
        ax.set_ylabel("Probability density")
        ax.set_title("Fitted normal distribution")
        ax.grid(True, axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.caption(
            "This is a fitted normal model. Patient counts are discrete, "
            "so the curve approximates their distribution rather than "
            "giving exact probabilities for individual counts."
        )

        st.divider()
        st.markdown("### Probability calculator")

        probability_type = st.selectbox(
            "Choose the probability you want to calculate",
            [
                "At most a given number of patients",
                "At least a given number of patients",
                "Between two patient counts",
            ],
        )

        lower = max(0, int(round(mean_arrivals)))
        upper = lower + 10

        if probability_type == "At most a given number of patients":
            k = st.number_input(
                "Patient count",
                min_value=0,
                max_value=10000,
                value=lower,
                step=1,
                key="normal_at_most",
            )
            probability = stats.norm.cdf(
                k + 0.5, loc=mean_arrivals, scale=std_arrivals
            )
            st.metric("Estimated probability", f"{probability:.2%}")
            st.write(
                f"Estimated probability of at most {k} arrivals: "
                f"{probability:.2%}."
            )

        elif probability_type == "At least a given number of patients":
            k = st.number_input(
                "Patient count",
                min_value=0,
                max_value=10000,
                value=lower,
                step=1,
                key="normal_at_least",
            )
            probability = stats.norm.sf(
                k - 0.5, loc=mean_arrivals, scale=std_arrivals
            )
            st.metric("Estimated probability", f"{probability:.2%}")
            st.write(
                f"Estimated probability of at least {k} arrivals: "
                f"{probability:.2%}."
            )

        else:
            col1, col2 = st.columns(2)
            with col1:
                low = st.number_input(
                    "Lower patient count",
                    min_value=0,
                    max_value=10000,
                    value=lower,
                    step=1,
                    key="normal_between_low",
                )
            with col2:
                high = st.number_input(
                    "Upper patient count",
                    min_value=0,
                    max_value=10000,
                    value=upper,
                    step=1,
                    key="normal_between_high",
                )

            if low > high:
                st.warning("The lower count must not exceed the upper count.")
            else:
                probability = (
                    stats.norm.cdf(
                        high + 0.5,
                        loc=mean_arrivals,
                        scale=std_arrivals,
                    )
                    - stats.norm.cdf(
                        low - 0.5,
                        loc=mean_arrivals,
                        scale=std_arrivals,
                    )
                )
                probability = float(np.clip(probability, 0.0, 1.0))
                st.metric("Estimated probability", f"{probability:.2%}")
                st.write(
                    f"Estimated probability of between {low} and {high} "
                    f"arrivals, inclusive: {probability:.2%}."
                )


                
# ---------------- PAGE: CONFIDENCE INTERVAL ----------------
elif page == "Confidence Interval":
    st.subheader("Confidence Interval for Average Patient Arrivals")
    st.write(
        "Estimate the mean daily arrivals and quantify uncertainty "
        "using a Student's t confidence interval."
    )

    arrivals = filtered["arrivals"].dropna()
    n = len(arrivals)

    if n < 2:
        st.warning("Select at least two days of valid arrival data.")
    else:
        sample_mean = float(arrivals.mean())
        sample_std = float(arrivals.std(ddof=1))
        standard_error = sample_std / np.sqrt(n)

        confidence_level = st.selectbox(
            "Confidence level",
            [90, 95, 99],
            index=1,
            format_func=lambda value: f"{value}% confidence",
        )

        alpha = 1 - confidence_level / 100
        critical_t = stats.t.ppf(1 - alpha / 2, df=n - 1)
        margin_error = critical_t * standard_error
        lower_bound = sample_mean - margin_error
        upper_bound = sample_mean + margin_error

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Sample mean", f"{sample_mean:.2f}")
        c2.metric("Lower confidence limit", f"{lower_bound:.2f}")
        c3.metric("Upper confidence limit", f"{upper_bound:.2f}")
        c4.metric("Margin of error", f"±{margin_error:.2f}")

        st.divider()
        st.markdown("### Confidence interval visualisation")

        fig, ax = plt.subplots(figsize=(10, 2.8))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        ax.errorbar(
            sample_mean,
            0,
            xerr=margin_error,
            fmt="o",
            color=BLUE,
            ecolor="#79A9D8",
            elinewidth=3,
            capsize=10,
            markersize=8,
        )
        ax.axvline(sample_mean, color=DARK_BLUE, linestyle="--", alpha=0.7)
        ax.set_yticks([])
        ax.set_xlabel("Average daily patient arrivals")
        ax.set_title(f"{confidence_level}% confidence interval")
        ax.grid(True, axis="x", alpha=0.25)
        ax.spines[["top", "right", "left"]].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.info(
            f"The estimated mean is {sample_mean:.2f} arrivals per day. "
            f"The {confidence_level}% confidence interval is "
            f"{lower_bound:.2f} to {upper_bound:.2f} arrivals per day."
        )

        st.caption(
            "Interpretation: the interval is produced by a method that, "
            "under its assumptions, captures the population mean at the "
            "selected confidence level in repeated sampling. It does not "
            "mean that the true mean has this probability of lying in this "
            "particular calculated interval."
        )

        st.warning(
            "The dataset is simulated. This interval demonstrates statistical "
            "estimation and should not be treated as a real hospital estimate."
        )

        
# ---------------- PAGE: SAMPLING AND CENTRAL LIMIT THEOREM ----------------
elif page == "Sampling & CLT":
    st.subheader("Sampling and Central Limit Theorem")
    st.write(
        "Repeatedly draw samples from the selected daily-arrival data "
        "and study the distribution of their sample means."
    )

    arrivals = filtered["arrivals"].dropna().to_numpy(dtype=float)
    n_population = len(arrivals)

    if n_population < 2:
        st.warning("Select at least two valid daily observations.")
    else:
        st.markdown("### Simulation settings")

        sample_size = st.slider(
            "Number of days in each sample",
            min_value=2,
            max_value=min(100, n_population),
            value=min(30, n_population),
            step=1,
        )

        number_simulations = st.slider(
            "Number of repeated samples",
            min_value=100,
            max_value=5000,
            value=1000,
            step=100,
        )

        if st.button("Run CLT simulation", type="primary"):
            rng = np.random.default_rng()
            sample_means = np.array([
                rng.choice(arrivals, size=sample_size, replace=True).mean()
                for _ in range(number_simulations)
            ])
            st.session_state["clt_sample_means"] = sample_means
            st.session_state["clt_sample_size"] = sample_size
            st.session_state["clt_number_simulations"] = number_simulations
            st.session_state["clt_population_mean"] = float(np.mean(arrivals))
            st.session_state["clt_population_std"] = float(np.std(arrivals, ddof=1))

        if "clt_sample_means" in st.session_state:
            sample_means = st.session_state["clt_sample_means"]
            saved_size = st.session_state["clt_sample_size"]
            saved_simulations = st.session_state["clt_number_simulations"]
            population_mean = st.session_state["clt_population_mean"]
            population_std = st.session_state["clt_population_std"]

            # Re-run if the controls have changed since the last simulation.
            if saved_size != sample_size or saved_simulations != number_simulations:
                st.info("Settings changed. Click 'Run CLT simulation' to update the results.")
            else:
                standard_error = population_std / np.sqrt(sample_size)

                st.divider()
                st.markdown("### Results")

                c1, c2, c3 = st.columns(3)
                c1.metric("Original data mean", f"{population_mean:.2f}")
                c2.metric("Mean of sample means", f"{np.mean(sample_means):.2f}")
                c3.metric("Standard deviation of sample means", f"{np.std(sample_means, ddof=1):.2f}")

                st.markdown("### Distribution of sample means")

                fig, ax = plt.subplots(figsize=(10, 4.5))
                fig.patch.set_facecolor("white")
                ax.set_facecolor("white")

                ax.hist(
                    sample_means,
                    bins="auto",
                    density=True,
                    color="#79A9D8",
                    edgecolor="white",
                    alpha=0.85,
                    label="Simulated sample means",
                )

                x_min, x_max = ax.get_xlim()
                x = np.linspace(x_min, x_max, 400)

                if standard_error > 0:
                    theoretical_density = stats.norm.pdf(
                        x,
                        loc=population_mean,
                        scale=standard_error,
                    )
                    ax.plot(
                        x,
                        theoretical_density,
                        color=DARK_BLUE,
                        linewidth=2.5,
                        label="Normal approximation",
                    )

                ax.axvline(
                    population_mean,
                    color="#C47A18",
                    linestyle="--",
                    linewidth=2,
                    label=f"Original mean = {population_mean:.2f}",
                )

                ax.set_xlabel("Sample mean of daily arrivals")
                ax.set_ylabel("Density")
                ax.grid(True, axis="y", alpha=0.25)
                ax.spines[["top", "right"]].set_visible(False)
                ax.legend(frameon=False)
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

                st.markdown("### What does this tell us?")
                st.write(
                    f"- Each simulated sample contains **{sample_size} days**."
                )
                st.write(
                    f"- The simulation generated **{number_simulations} sample means**."
                )
                st.write(
                    f"- The theoretical standard error is approximately "
                    f"**{standard_error:.2f} arrivals**."
                )
                st.write(
                    "- As sample size increases, the sample means generally "
                    "vary less and their distribution often becomes more "
                    "nearly normal, under the Central Limit Theorem's conditions."
                )

                st.caption(
                    "This simulation samples with replacement from the selected "
                    "empirical daily counts. The normal curve is an approximation. "
                    "The CLT does not guarantee that the original daily counts "
                    "themselves are normally distributed."
                )

        else:
            st.info("Choose your settings and click 'Run CLT simulation' to see the results.")

        
# ---------------- PAGE: ANOVA TEST ----------------
elif page == "ANOVA Test":
    st.subheader("One-Way ANOVA: Comparing Daily Arrivals by Weekday")

    st.write(
        "This test checks whether mean daily arrivals differ across "
        "the seven weekdays."
    )

    anova_data = filtered[["weekday", "arrivals"]].dropna().copy()
    weekday_order = [
        "Monday", "Tuesday", "Wednesday", "Thursday",
        "Friday", "Saturday", "Sunday"
    ]

    groups = [
        anova_data.loc[
            anova_data["weekday"] == day, "arrivals"
        ].astype(float).to_numpy()
        for day in weekday_order
    ]
    groups = [group for group in groups if len(group) >= 2]

    st.markdown("### Hypotheses")
    st.write("**H₀:** Mean daily arrivals are equal across all weekdays.")
    st.write("**H₁:** At least one weekday has a different mean.")

    if len(groups) < 2:
        st.warning("At least two weekdays with two or more observations are required.")
    else:
        f_stat, p_value = stats.f_oneway(*groups)

        summary = (
            anova_data.groupby("weekday")["arrivals"]
            .agg(["count", "mean", "std"])
            .reindex(weekday_order)
            .dropna()
            .reset_index()
        )

        st.markdown("### Average arrivals by weekday")
        st.dataframe(
            summary.rename(columns={
                "weekday": "Weekday",
                "count": "Number of days",
                "mean": "Mean arrivals",
                "std": "Standard deviation",
            }).round(2),
            use_container_width=True,
            hide_index=True,
        )

        c1, c2 = st.columns(2)
        c1.metric("F-statistic", f"{f_stat:.3f}")
        c2.metric("p-value", f"{p_value:.5f}")

        alpha = st.select_slider(
            "Significance level (α)",
            options=[0.01, 0.05, 0.10],
            value=0.05,
            key="anova_alpha",
        )

        if p_value < alpha:
            st.warning(
                "Reject H₀: there is evidence that not all weekday "
                "mean arrivals are equal."
            )
        else:
            st.success(
                "Fail to reject H₀: there is insufficient evidence "
                "of a difference in mean arrivals across weekdays."
            )

        st.markdown("### Distribution by weekday")
        fig, ax = plt.subplots(figsize=(10, 4.5))
        plot_data = [
            anova_data.loc[
                anova_data["weekday"] == day, "arrivals"
            ].to_numpy()
            for day in weekday_order
        ]
        plot_labels = [
            day for day, values in zip(weekday_order, plot_data)
            if len(values) > 0
        ]
        plot_data = [values for values in plot_data if len(values) > 0]

        ax.boxplot(plot_data, tick_labels=plot_labels)
        ax.set_xlabel("Weekday")
        ax.set_ylabel("Daily arrivals")
        ax.grid(True, axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        plt.xticks(rotation=20)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.caption(
            "ANOVA assumes independent observations and approximately "
            "normal within-group errors with similar variances. "
            "A significant result does not identify which specific "
            "weekdays differ. These data are simulated."
        )

        
# ---------------- PAGE: LEAST-SQUARES CURVE FITTING ----------------
elif page == "Least-Squares Curve Fitting":
    st.subheader("Least-Squares Curve Fitting")

    st.write(
        "Fit a straight line to daily arrivals by minimizing the "
        "sum of squared differences between observed and predicted arrivals."
    )

    curve_data = filtered[["date", "arrivals"]].dropna().copy()
    curve_data["date"] = pd.to_datetime(curve_data["date"])
    curve_data = curve_data.sort_values("date").reset_index(drop=True)

    if len(curve_data) < 2:
        st.warning("Select at least two valid daily observations.")
    else:
        x = np.arange(len(curve_data), dtype=float)
        y = curve_data["arrivals"].to_numpy(dtype=float)

        slope, intercept = np.polyfit(x, y, 1)
        fitted = slope * x + intercept

        residuals = y - fitted
        sse = float(np.sum(residuals ** 2))
        mae = float(np.mean(np.abs(residuals)))
        rmse = float(np.sqrt(np.mean(residuals ** 2)))

        ss_total = float(np.sum((y - np.mean(y)) ** 2))
        r_squared = (
            1 - sse / ss_total if ss_total > 0 else float("nan")
        )

        st.markdown("### Fitted line")
        st.latex(
            rf"\\hat{{y}} = {slope:.3f}x + {intercept:.3f}"
        )
        st.write(
            "Here, x is the observation's position in the selected date "
            "range, and ŷ is the fitted daily-arrival count."
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("MAE", f"{mae:.2f}")
        c2.metric("RMSE", f"{rmse:.2f}")
        c3.metric("R²", f"{r_squared:.3f}")

        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.scatter(
            curve_data["date"], y,
            alpha=0.65, label="Observed arrivals"
        )
        ax.plot(
            curve_data["date"], fitted,
            color=DARK_BLUE, linewidth=2.5,
            label="Least-squares line"
        )
        ax.set_xlabel("Date")
        ax.set_ylabel("Daily arrivals")
        ax.legend(frameon=False)
        ax.grid(True, axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        fig.autofmt_xdate()
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        if slope > 0:
            st.write(
                f"The fitted trend increases by about {slope:.2f} "
                "arrivals per observation."
            )
        elif slope < 0:
            st.write(
                f"The fitted trend decreases by about {abs(slope):.2f} "
                "arrivals per observation."
            )
        else:
            st.write("The fitted line has no overall linear trend.")

        st.caption(
            "Least squares describes the linear trend"
        )



# ---------------- PAGE: LIVE PATIENT ARRIVALS ----------------
elif page == "Live Patient Arrivals":
    import json
    from streamlit_autorefresh import st_autorefresh
    from urllib.request import urlopen
    from urllib.error import URLError, HTTPError

    st.subheader("Live Patient Arrivals")
    st_autorefresh(interval=5000, key="live_arrivals_refresh")
    st.write(
        "View patient-arrival records received by the FastAPI backend. "
        "This page is separate from the historical statistical analysis."
    )

    if st.button("Refresh live data"):
        st.rerun()

    api_url = "http://127.0.0.1:8000/api/v1/arrivals"
    summary_url = "http://127.0.0.1:8000/api/v1/summary"

    try:
        with urlopen(api_url, timeout=5) as response:
            result = json.loads(response.read().decode("utf-8"))

        records = result.get("records", [])

        if not records:
            st.info(
                "The API is connected, but no arrival records were found. "
                "Start the simulation to send records."
            )
        else:
            st.success(f"Connected to API — {len(records)} records retrieved.")

            departments = sorted({
                record.get("department", "Unknown")
                for record in records
            })
            selected_department = st.selectbox(
                "Filter by department",
                ["All departments"] + departments
            )

            if selected_department != "All departments":
                records = [
                    record for record in records
                    if record.get("department") == selected_department
                ]

            if records:
                counts = [
                    int(record.get("patient_count", 0))
                    for record in records
                ]

                col1, col2, col3 = st.columns(3)
                col1.metric("Records shown", len(records))
                col2.metric("Total patients in records", sum(counts))
                col3.metric("Average per record", f"{sum(counts) / len(counts):.2f}")

                st.markdown("### Latest arrival records")
                live_df = pd.DataFrame(records)

                if "period_start" in live_df.columns:
                    live_df["period_start"] = pd.to_datetime(
                        live_df["period_start"], errors="coerce"
                    )

                st.dataframe(live_df, use_container_width=True)

                if "period_start" in live_df.columns and "patient_count" in live_df.columns:
                    chart_df = live_df.dropna(subset=["period_start"]).sort_values(
                        "period_start"
                    )
                    if not chart_df.empty:
                        st.markdown("### Patient counts over time")
                        chart_df = chart_df.set_index("period_start")
                        st.line_chart(chart_df["patient_count"])

                st.caption(
                    "These are incoming API records. The figures above summarize "
                    "the records currently returned by the API, not necessarily "
                    "unique patients."
                )

    except (HTTPError, URLError, TimeoutError, OSError) as error:
        st.error(
            "Could not connect to the FastAPI backend. Make sure the backend "
            "is running at http://127.0.0.1:8000."
        )
        st.caption(f"Connection detail: {error}")
    except (json.JSONDecodeError, ValueError, TypeError) as error:
        st.error("The API returned data that could not be displayed.")
        st.caption(f"Data detail: {error}")


