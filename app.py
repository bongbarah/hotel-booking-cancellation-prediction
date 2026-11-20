
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from config import (
    CLEAN_FILE, MODEL_FILE, METRICS_FILE, FIGURES, RESULTS,
    TARGET, FEATURES, NUMERIC, CATEGORICAL
)


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Hotel Intelligence | Revenue & Risk",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 2. VISUAL DESIGN
# ============================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background-color: #F4F6F9;
}

[data-testid="stSidebar"] {
    background-color: #101D30;
}

[data-testid="stSidebar"] * {
    color: #E8EEF5;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label {
    color: #E8EEF5;
}

.main-title {
    font-family: 'Manrope', sans-serif;
    font-size: 34px;
    font-weight: 800;
    color: #14243A;
    margin-bottom: 3px;
}

.subtitle {
    color: #68788C;
    font-size: 14px;
    margin-bottom: 25px;
}

.section-title {
    font-family: 'Manrope', sans-serif;
    font-size: 23px;
    font-weight: 750;
    color: #14243A;
    margin-top: 15px;
}

.section-description {
    color: #718096;
    font-size: 14px;
    margin-bottom: 18px;
}

.story-card {
    background-color: #FFFFFF;
    padding: 19px;
    border-radius: 12px;
    border: 1px solid #E4E9F0;
    margin-bottom: 12px;
}

.story-label {
    color: #138C8C;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
}

.story-heading {
    color: #172A43;
    font-size: 17px;
    font-weight: 700;
    margin-top: 6px;
}

.story-text {
    color: #64748B;
    font-size: 14px;
    margin-top: 7px;
    line-height: 1.6;
}

div[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #E5EAF0;
    padding: 17px;
    border-radius: 12px;
}

div[data-testid="stMetricLabel"] {
    color: #66758A;
}

div[data-testid="stMetricValue"] {
    color: #14243A;
    font-family: 'Manrope', sans-serif;
    font-weight: 800;
}

div[data-testid="stTabs"] button {
    font-weight: 600;
}

hr {
    border-color: #E3E8EF;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# 3. LOAD PROJECT DATA
# ============================================================

@st.cache_data
def load_data():
    return pd.read_csv(CLEAN_FILE)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)


@st.cache_data
def load_metrics():
    return pd.read_csv(METRICS_FILE)


if not CLEAN_FILE.exists():
    st.error("Cleaned dataset not found. Run python run_pipeline.py first.")
    st.stop()

if not MODEL_FILE.exists():
    st.error("Trained model not found. Run python run_pipeline.py first.")
    st.stop()

if not METRICS_FILE.exists():
    st.error("Model comparison results not found. Run python run_pipeline.py first.")
    st.stop()


df = load_data()
bundle = load_model()
metrics = load_metrics()

model = bundle["model"]
model_name = bundle["model_name"]
threshold = float(bundle.get("threshold", 0.5))


# Validate the expected model features.

missing_features = [c for c in FEATURES if c not in df.columns]

if missing_features:
    st.error(f"Missing model features: {missing_features}")
    st.stop()

if TARGET not in df.columns:
    st.error(f"Target column {TARGET} is missing.")
    st.stop()


# ============================================================
# 4. HELPER FUNCTIONS
# ============================================================

MONTH_ORDER = [
    "January", "February", "March", "April",
    "May", "June", "July", "August",
    "September", "October", "November", "December"
]

COLORS = {
    "navy": "#14243A",
    "teal": "#138C8C",
    "gold": "#D6A64F",
    "red": "#D95D5D",
    "blue": "#5479A5",
    "grey": "#8291A5"
}


def money(value):
    return f"{value:,.2f}"


def percentage(value):
    return f"{value:.1%}"


def add_story(label, heading, text):
    st.markdown(
        f"""
        <div class="story-card">
            <div class="story-label">{label}</div>
            <div class="story-heading">{heading}</div>
            <div class="story-text">{text}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def cancellation_rate(data):
    if len(data) == 0:
        return 0
    return float(data[TARGET].mean())


def grouped_cancellation(data, column):
    result = (
        data.groupby(column, dropna=False)[TARGET]
        .agg(["mean", "count"])
        .reset_index()
    )

    result["cancellation_rate"] = result["mean"] * 100

    return result


def create_chart(fig, height=400):
    fig.update_layout(
        height=height,
        template="plotly_white",
        font=dict(family="DM Sans", color=COLORS["navy"]),
        title_font=dict(size=17),
        margin=dict(l=15, r=15, t=55, b=20),
        legend_title_text=""
    )

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#E9EEF3")

    st.plotly_chart(fig, width="stretch")


# ============================================================
# 5. SIDEBAR NAVIGATION AND FILTERS
# ============================================================

with st.sidebar:

    st.markdown("## 🏨 HOTEL INTELLIGENCE")
    st.caption("REVENUE & CANCELLATION ANALYTICS")

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Executive Overview",
            "Booking Behaviour",
            "Cancellation Intelligence",
            "Predictive Performance",
            "Booking Risk Assessment",
            "Revenue Intelligence"
        ]
    )

    st.divider()

    st.markdown("### ANALYSIS FILTERS")

    hotel_options = sorted(df["hotel"].dropna().unique().tolist())

    selected_hotels = st.multiselect(
        "Hotel type",
        hotel_options,
        default=hotel_options
    )

    selected_months = st.multiselect(
        "Arrival month",
        MONTH_ORDER,
        default=MONTH_ORDER
    )

    segment_options = sorted(
        df["market_segment"].dropna().unique().tolist()
    )

    selected_segments = st.multiselect(
        "Market segment",
        segment_options,
        default=segment_options
    )

    st.divider()

    st.caption("PROJECT INFORMATION")
    st.caption("Dataset: Hotel Booking Demand")
    st.caption(f"Records: {len(df):,}")
    st.caption(f"Selected model: {model_name}")
    st.caption("Prediction scenario: Booking creation")


# Apply filters to analytical pages.

filtered = df[
    df["hotel"].isin(selected_hotels) &
    df["arrival_date_month"].isin(selected_months) &
    df["market_segment"].isin(selected_segments)
].copy()


# ============================================================
# 6. HEADER
# ============================================================

st.markdown(
    '<div class="main-title">Hotel Revenue & Cancellation Intelligence</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Understanding booking behaviour, predicting cancellation risk and exploring revenue exposure.</div>',
    unsafe_allow_html=True
)

if filtered.empty:
    st.warning("No records match the selected filters. Please adjust the filters.")
    st.stop()


# ============================================================
# PAGE 1: EXECUTIVE OVERVIEW
# ============================================================

if page == "Executive Overview":

    st.markdown("## Executive Overview")
    st.caption(
        "A management-level snapshot of booking performance and cancellation patterns."
    )

    total = len(filtered)
    cancelled = int(filtered[TARGET].sum())
    completed = total - cancelled
    rate = cancellation_rate(filtered)

    estimated_value = (
        filtered["adr"] * filtered["total_nights"]
    ).sum()

    cancelled_value = (
        filtered.loc[filtered[TARGET] == 1, "adr"] *
        filtered.loc[filtered[TARGET] == 1, "total_nights"]
    ).sum()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total bookings", f"{total:,}")
    col2.metric("Cancelled bookings", f"{cancelled:,}")
    col3.metric("Cancellation rate", percentage(rate))
    col4.metric("Selected model", model_name)

    st.markdown("---")

    left, right = st.columns([1.5, 1])

    with left:

        hotel_analysis = grouped_cancellation(filtered, "hotel")

        fig = px.bar(
            hotel_analysis,
            x="hotel",
            y="cancellation_rate",
            color="hotel",
            text="cancellation_rate",
            title="How does cancellation performance differ by hotel type?",
            labels={
                "hotel": "Hotel type",
                "cancellation_rate": "Cancellation rate (%)"
            },
            color_discrete_sequence=[
                COLORS["teal"], COLORS["gold"]
            ]
        )

        fig.update_traces(
            texttemplate="%{text:.1f}%",
            textposition="outside"
        )

        create_chart(fig)

    with right:

        outcome = pd.DataFrame({
            "Booking outcome": ["Not cancelled", "Cancelled"],
            "Bookings": [completed, cancelled]
        })

        fig = px.pie(
            outcome,
            names="Booking outcome",
            values="Bookings",
            hole=0.65,
            title="Booking outcome distribution",
            color="Booking outcome",
            color_discrete_map={
                "Not cancelled": COLORS["teal"],
                "Cancelled": COLORS["red"]
            }
        )

        create_chart(fig)

    st.markdown("### What should management pay attention to?")

    highest_hotel = hotel_analysis.sort_values(
        "cancellation_rate", ascending=False
    ).iloc[0]

    add_story(
        "KEY FINDING",
        "Cancellation performance varies by hotel type",
        f"{highest_hotel['hotel']} has the highest cancellation rate "
        f"in the selected data, at {highest_hotel['cancellation_rate']:.1f}%. "
        "This identifies an area for further investigation."
    )

    add_story(
        "REVENUE CONTEXT",
        "Cancelled bookings represent estimated booking-value exposure",
        f"The ADR multiplied by length-of-stay proxy for cancelled bookings "
        f"is {money(cancelled_value)}. This is an estimate, not confirmed lost revenue."
    )

    add_story(
        "BUSINESS QUESTION",
        "Which booking characteristics are associated with cancellations?",
        "The following sections investigate lead time, customer segments, "
        "distribution channels, deposit types and seasonal patterns."
    )


# ============================================================
# PAGE 2: BOOKING BEHAVIOUR
# ============================================================

elif page == "Booking Behaviour":

    st.markdown("## Booking Behaviour")
    st.caption(
        "Exploring how customers book, when they arrive and which channels they use."
    )

    st.markdown("### Question 1: Which customer segments generate the most bookings?")

    segment_counts = (
        filtered["market_segment"]
        .value_counts()
        .reset_index()
    )

    segment_counts.columns = ["Market segment", "Bookings"]

    fig = px.bar(
        segment_counts,
        x="Market segment",
        y="Bookings",
        color="Bookings",
        title="Bookings by market segment",
        color_continuous_scale="Teal"
    )

    create_chart(fig)

    largest_segment = segment_counts.iloc[0]

    add_story(
        "WHAT THE DATA SHOWS",
        f"{largest_segment['Market segment']} is the largest booking segment",
        f"It accounts for {largest_segment['Bookings']:,} bookings "
        f"within the selected dataset."
    )

    st.markdown("### Question 2: How do booking lead times vary?")

    fig = px.histogram(
        filtered,
        x="lead_time",
        color=filtered[TARGET].map({
            0: "Not cancelled",
            1: "Cancelled"
        }),
        nbins=50,
        barmode="overlay",
        opacity=0.7,
        title="Distribution of booking lead times",
        labels={
            "lead_time": "Lead time (days)",
            "color": "Booking outcome"
        },
        color_discrete_map={
            "Not cancelled": COLORS["teal"],
            "Cancelled": COLORS["red"]
        }
    )

    create_chart(fig)

    st.markdown("### Question 3: When do customers arrive?")

    monthly = filtered.copy()

    monthly["arrival_date_month"] = pd.Categorical(
        monthly["arrival_date_month"],
        categories=MONTH_ORDER,
        ordered=True
    )

    monthly = (
        monthly.groupby("arrival_date_month", observed=False)
        .size()
        .reset_index(name="Bookings")
    )

    fig = px.line(
        monthly,
        x="arrival_date_month",
        y="Bookings",
        markers=True,
        title="Booking arrival patterns by month"
    )

    fig.update_traces(line_color=COLORS["teal"])

    create_chart(fig)

    st.markdown("### Question 4: Which distribution channels are used?")

    channel = (
        filtered["distribution_channel"]
        .value_counts()
        .reset_index()
    )

    channel.columns = ["Distribution channel", "Bookings"]

    fig = px.bar(
        channel,
        x="Distribution channel",
        y="Bookings",
        color="Distribution channel",
        title="Bookings by distribution channel"
    )

    create_chart(fig)


# ============================================================
# PAGE 3: CANCELLATION INTELLIGENCE
# ============================================================

elif page == "Cancellation Intelligence":

    st.markdown("## Cancellation Intelligence")
    st.caption(
        "Investigating the characteristics associated with booking cancellations."
    )

    st.markdown("### Question 1: Does lead time relate to cancellation risk?")

    lead_data = filtered.copy()

    lead_data["lead_time_group"] = pd.cut(
        lead_data["lead_time"],
        bins=[-1, 7, 30, 90, 180, 365, np.inf],
        labels=[
            "0–7 days",
            "8–30 days",
            "31–90 days",
            "91–180 days",
            "181–365 days",
            "365+ days"
        ]
    )

    lead_analysis = grouped_cancellation(
        lead_data, "lead_time_group"
    )

    fig = px.bar(
        lead_analysis,
        x="lead_time_group",
        y="cancellation_rate",
        color="cancellation_rate",
        title="Cancellation rate by booking lead time",
        labels={
            "lead_time_group": "Lead-time group",
            "cancellation_rate": "Cancellation rate (%)"
        },
        color_continuous_scale="Teal"
    )

    create_chart(fig)

    highest_lead = lead_analysis.sort_values(
        "cancellation_rate", ascending=False
    ).iloc[0]

    add_story(
        "KEY FINDING",
        f"{highest_lead['lead_time_group']} has the highest observed cancellation rate",
        f"The cancellation rate for this group is "
        f"{highest_lead['cancellation_rate']:.1f}%. "
        "This is an observed association, not proof that lead time causes cancellation."
    )

    st.markdown("### Question 2: Does deposit type matter?")

    deposit_analysis = grouped_cancellation(
        filtered, "deposit_type"
    ).sort_values("cancellation_rate", ascending=False)

    fig = px.bar(
        deposit_analysis,
        x="deposit_type",
        y="cancellation_rate",
        color="deposit_type",
        title="Cancellation rate by deposit type",
        labels={
            "deposit_type": "Deposit type",
            "cancellation_rate": "Cancellation rate (%)"
        }
    )

    create_chart(fig)

    st.markdown("### Question 3: Which customer types have higher cancellation rates?")

    customer_analysis = grouped_cancellation(
        filtered, "customer_type"
    ).sort_values("cancellation_rate", ascending=False)

    fig = px.bar(
        customer_analysis,
        x="customer_type",
        y="cancellation_rate",
        color="customer_type",
        title="Cancellation rate by customer type",
        labels={
            "customer_type": "Customer type",
            "cancellation_rate": "Cancellation rate (%)"
        }
    )

    create_chart(fig)

    st.markdown("### Question 4: Are cancellations different across market segments?")

    segment_analysis = grouped_cancellation(
        filtered, "market_segment"
    ).sort_values("cancellation_rate", ascending=False)

    fig = px.bar(
        segment_analysis,
        x="market_segment",
        y="cancellation_rate",
        color="cancellation_rate",
        title="Cancellation rate by market segment",
        labels={
            "market_segment": "Market segment",
            "cancellation_rate": "Cancellation rate (%)"
        },
        color_continuous_scale="Teal"
    )

    create_chart(fig)

    st.markdown("### Question 5: How does the average daily rate differ by outcome?")

    adr_analysis = (
        filtered.groupby(TARGET)["adr"]
        .mean()
        .reset_index()
    )

    adr_analysis["Outcome"] = adr_analysis[TARGET].map({
        0: "Not cancelled",
        1: "Cancelled"
    })

    fig = px.bar(
        adr_analysis,
        x="Outcome",
        y="adr",
        color="Outcome",
        title="Average daily rate by booking outcome",
        color_discrete_map={
            "Not cancelled": COLORS["teal"],
            "Cancelled": COLORS["red"]
        }
    )

    create_chart(fig)

    st.caption(
        "These comparisons describe historical associations. "
        "They do not establish causation."
    )



# ============================================================
# PAGE 4: PREDICTIVE PERFORMANCE
# ============================================================

elif page == "Predictive Performance":

    st.markdown("## Predictive Performance")
    st.caption(
        "Evaluating cancellation prediction performance using five-fold "
        "group-aware cross-validation."
    )

    # --------------------------------------------------------
    # Model comparison
    # --------------------------------------------------------

    st.markdown("### Model comparison")

    display_metrics = metrics.copy()

    st.dataframe(
        display_metrics.style.format({
            "accuracy": "{:.3f}",
            "precision": "{:.3f}",
            "recall": "{:.3f}",
            "f1": "{:.3f}",
            "roc_auc": "{:.3f}",
            "pr_auc": "{:.3f}"
        }),
        width="stretch",
        hide_index=True
    )

    best_row = display_metrics.sort_values(
        "pr_auc", ascending=False
    ).iloc[0]

    col1, col2, col3 = st.columns(3)

    col1.metric("Selected model", model_name)
    col2.metric("Best PR-AUC", f"{best_row['pr_auc']:.3f}")
    col3.metric("Prediction threshold", f"{threshold:.2f}")

    st.markdown("### Model comparison using PR-AUC")

    fig = px.bar(
        display_metrics.sort_values(
            "pr_auc", ascending=True
        ),
        x="pr_auc",
        y="model",
        orientation="h",
        text="pr_auc",
        color="pr_auc",
        color_continuous_scale="Teal",
        title="Five-fold out-of-fold model performance",
        labels={
            "pr_auc": "PR-AUC",
            "model": "Model"
        }
    )

    fig.update_traces(
        texttemplate="%{text:.3f}"
    )

    create_chart(fig)

    add_story(
        "MODEL SELECTION",
        f"{model_name} achieved the highest PR-AUC",
        f"Its five-fold out-of-fold PR-AUC is "
        f"{best_row['pr_auc']:.3f}. "
        "The model was selected using PR-AUC."
    )

    st.markdown("---")

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    st.markdown("### Confusion matrix")

    confusion_path = (
        FIGURES / "models" / "confusion_matrix.png"
    )

    if confusion_path.exists():

        st.image(
            str(confusion_path),
            caption=(
                "Out-of-fold confusion matrix for the selected model "
                "at a probability threshold of 0.50."
            ),
            width="stretch"
        )

    else:
        st.info("Confusion matrix has not been generated.")

    st.caption(
        "The confusion matrix is based on out-of-fold predictions. "
        "Each record was evaluated by a model that did not train on that record."
    )

    st.markdown("---")

    # --------------------------------------------------------
    # Permutation feature importance
    # --------------------------------------------------------

    st.markdown("### What influences cancellation predictions?")

    st.markdown("#### Permutation feature importance")

    permutation_path = (
        FIGURES / "explainability" /
        "permutation_importance.png"
    )

    permutation_csv = (
        RESULTS / "permutation_importance.csv"
    )

    if permutation_path.exists():

        st.image(
            str(permutation_path),
            caption=(
                "Permutation importance measured using average precision "
                "on a held-out validation fold."
            ),
            width="stretch"
        )

    else:
        st.info("Permutation importance chart is not available.")

    if permutation_csv.exists():

        permutation_table = pd.read_csv(
            permutation_csv
        )

        st.markdown("**Top 10 original booking features**")

        st.dataframe(
            permutation_table.head(10).style.format({
                "importance": "{:.4f}",
                "std": "{:.4f}"
            }),
            width="stretch",
            hide_index=True
        )

    st.markdown("---")

    # --------------------------------------------------------
    # SHAP explainability
    # --------------------------------------------------------

    st.markdown("### SHAP model explainability")

    st.write(
        "SHAP helps explain how individual features contribute "
        "to the model's predictions. The following charts show "
        "global feature influence across a sample of validation bookings."
    )

    shap_summary_path = (
        FIGURES / "explainability" /
        "shap_summary.png"
    )

    shap_importance_path = (
        FIGURES / "explainability" /
        "shap_importance.png"
    )

    shap_csv = (
        RESULTS / "shap_feature_importance.csv"
    )

    if shap_summary_path.exists():

        st.markdown("#### SHAP summary plot")

        st.image(
            str(shap_summary_path),
            caption=(
                "SHAP summary showing the direction and magnitude "
                "of feature contributions."
            ),
            width="stretch"
        )

    else:
        st.info("SHAP summary plot is not available.")

    if shap_importance_path.exists():

        st.markdown("#### SHAP global feature importance")

        st.image(
            str(shap_importance_path),
            caption=(
                "Features ranked by their mean absolute SHAP values."
            ),
            width="stretch"
        )

    if shap_csv.exists():

        shap_table = pd.read_csv(
            shap_csv
        )

        st.markdown("#### Top 10 SHAP features")

        st.dataframe(
            shap_table.head(10).style.format({
                "mean_absolute_shap": "{:.4f}"
            }),
            width="stretch",
            hide_index=True
        )

    st.warning(
        "Feature importance and SHAP values describe how the model "
        "uses information to make predictions. They do not establish "
        "causation. SHAP values are calculated on a held-out validation "
        "sample, while permutation importance is measured on a held-out fold."
    )

    st.info(
        "Validation note: the reported metrics use five-fold "
        "StratifiedGroupKFold out-of-fold predictions. Identical "
        "predictor profiles are kept together within the same fold. "
        "The final selected model is subsequently retrained on the "
        "full cleaned dataset."
    )


# ============================================================
# PAGE 5: BOOKING RISK ASSESSMENT
# ============================================================

elif page == "Booking Risk Assessment":

    st.markdown("## Booking Risk Assessment")
    st.caption(
        "Estimate cancellation probability using information available when a booking is created."
    )

    add_story(
        "PREDICTION SCENARIO",
        "Assess a booking before its arrival date",
        "Enter booking details to estimate its cancellation probability. "
        "The prediction is based on the currently trained baseline model."
    )

    with st.form("booking_prediction_form"):

        st.markdown("### Booking details")

        left, right = st.columns(2)

        with left:

            hotel = st.selectbox(
                "Hotel",
                sorted(df["hotel"].dropna().unique())
            )

            lead = st.number_input(
                "Lead time (days)", 0, 1000, 30
            )

            year = st.selectbox(
                "Arrival year",
                sorted(df["arrival_date_year"].dropna().unique())
            )

            month = st.selectbox(
                "Arrival month",
                MONTH_ORDER
            )

            week = st.number_input(
                "Arrival week", 1, 53, 20
            )

            day = st.number_input(
                "Arrival day", 1, 31, 15
            )

            weekend_nights = st.number_input(
                "Weekend nights", 0, 30, 1
            )

            week_nights = st.number_input(
                "Week nights", 0, 60, 2
            )

            adults = st.number_input(
                "Adults", 0, 20, 2
            )

            children = st.number_input(
                "Children", 0, 10, 0
            )

            babies = st.number_input(
                "Babies", 0, 10, 0
            )

        with right:

            meal = st.selectbox(
                "Meal",
                sorted(df["meal"].dropna().unique())
            )

            country = st.text_input(
                "Country code",
                "GBR"
            )

            market = st.selectbox(
                "Market segment",
                sorted(df["market_segment"].dropna().unique())
            )

            channel = st.selectbox(
                "Distribution channel",
                sorted(df["distribution_channel"].dropna().unique())
            )

            repeated = st.selectbox(
                "Repeated guest",
                [0, 1]
            )

            previous_cancel = st.number_input(
                "Previous cancellations", 0, 100, 0
            )

            previous_completed = st.number_input(
                "Previous completed bookings", 0, 100, 0
            )

            room = st.selectbox(
                "Reserved room type",
                sorted(df["reserved_room_type"].dropna().unique())
            )

            deposit = st.selectbox(
                "Deposit type",
                sorted(df["deposit_type"].dropna().unique())
            )

            agent = st.text_input(
                "Agent ID or Unknown",
                "Unknown"
            )

            customer = st.selectbox(
                "Customer type",
                sorted(df["customer_type"].dropna().unique())
            )

            adr = st.number_input(
                "Average daily rate (ADR)",
                min_value=0.0,
                max_value=1000.0,
                value=100.0
            )

        submitted = st.form_submit_button(
            "Calculate cancellation risk",
            type="primary",
            width="stretch"
        )

    if submitted:

        row = {
            "lead_time": lead,
            "arrival_date_year": year,
            "arrival_date_week_number": week,
            "arrival_date_day_of_month": day,
            "stays_in_weekend_nights": weekend_nights,
            "stays_in_week_nights": week_nights,
            "adults": adults,
            "children": children,
            "babies": babies,
            "is_repeated_guest": repeated,
            "previous_cancellations": previous_cancel,
            "previous_bookings_not_canceled": previous_completed,
            "adr": adr,
            "total_guests": adults + children + babies,
            "total_nights": weekend_nights + week_nights,
            "hotel": hotel,
            "arrival_date_month": month,
            "meal": meal,
            "country": country.strip().upper() or "Unknown",
            "market_segment": market,
            "distribution_channel": channel,
            "reserved_room_type": room,
            "deposit_type": deposit,
            "agent": agent.strip() or "Unknown",
            "customer_type": customer
        }

        if row["total_guests"] == 0:
            st.error("Please enter at least one guest.")
        elif row["total_nights"] == 0:
            st.error("Please enter at least one night.")
        else:

            input_data = pd.DataFrame(
                [row],
                columns=FEATURES
            )

            probability = float(
                model.predict_proba(input_data)[0, 1]
            )

            prediction = int(probability >= threshold)

            st.divider()

            st.markdown("### Prediction result")

            col1, col2 = st.columns(2)

            col1.metric(
                "Estimated cancellation probability",
                f"{probability:.1%}"
            )

            if prediction == 1:
                col2.metric("Model classification", "Higher risk")
                st.warning(
                    "The model classifies this booking as likely to cancel "
                    "at the current threshold."
                )
            else:
                col2.metric("Model classification", "Lower risk")
                st.success(
                    "The model classifies this booking as less likely to cancel "
                    "at the current threshold."
                )

            st.progress(probability)

            st.caption(
                f"Classification threshold: {threshold:.2f}. "
                "This is a model estimate, not a guarantee."
            )


# ============================================================
# PAGE 6: REVENUE INTELLIGENCE
# ============================================================

elif page == "Revenue Intelligence":

    st.markdown("## Revenue Intelligence")
    st.caption(
        "Exploring the estimated booking value associated with cancellation patterns."
    )

    revenue = filtered.copy()

    revenue["estimated_booking_value"] = (
        revenue["adr"] * revenue["total_nights"]
    )

    cancelled = revenue[revenue[TARGET] == 1].copy()

    total_cancelled_value = (
        cancelled["estimated_booking_value"].sum()
    )

    average_cancelled_value = (
        cancelled["estimated_booking_value"].mean()
        if len(cancelled) > 0 else 0
    )

    total_booked_value = (
        revenue["estimated_booking_value"].sum()
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Estimated value of cancelled bookings",
        money(total_cancelled_value)
    )

    col2.metric(
        "Average cancelled booking value",
        money(average_cancelled_value)
    )

    col3.metric(
        "Estimated value across all bookings",
        money(total_booked_value)
    )

    st.markdown("---")

    st.markdown(
        "### Question 1: Which hotel type has the greatest cancellation exposure?"
    )

    hotel_revenue = (
        cancelled.groupby("hotel")["estimated_booking_value"]
        .sum()
        .reset_index()
    )

    fig = px.bar(
        hotel_revenue,
        x="hotel",
        y="estimated_booking_value",
        color="hotel",
        title="Estimated booking value of cancelled reservations",
        labels={
            "hotel": "Hotel type",
            "estimated_booking_value": "Estimated booking value"
        }
    )

    create_chart(fig)

    st.markdown(
        "### Question 2: Which market segments contribute most to exposure?"
    )

    segment_revenue = (
        cancelled.groupby("market_segment")["estimated_booking_value"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )

    fig = px.bar(
        segment_revenue,
        x="market_segment",
        y="estimated_booking_value",
        color="estimated_booking_value",
        title="Estimated cancellation exposure by market segment",
        color_continuous_scale="Teal"
    )

    create_chart(fig)

    st.markdown(
        "### Question 3: Which cancelled bookings have the highest estimated value?"
    )

    top_cancelled = cancelled.nlargest(
        10, "estimated_booking_value"
    )

    st.dataframe(
        top_cancelled[
            [
                "hotel",
                "lead_time",
                "market_segment",
                "customer_type",
                "adr",
                "total_nights",
                "estimated_booking_value"
            ]
        ],
        width="stretch",
        hide_index=True
    )

    add_story(
        "FINANCIAL INTERPRETATION",
        "Booking value is a proxy for revenue exposure",
        "The calculation uses ADR multiplied by the number of nights. "
        "It does not account for replacement bookings, no-show fees, "
        "refunds, operational costs or actual realised revenue."
    )

    st.markdown("### Management considerations")

    st.markdown("""
    - Review booking segments with high cancellation exposure.
    - Investigate deposit policies and booking lead times.
    - Consider targeted booking confirmation strategies.
    - Monitor cancellation trends over time.
    - Evaluate whether interventions improve retained revenue.
    """)


# ============================================================
# 7. FOOTER
# ============================================================

st.divider()

st.caption(
    "Hotel Booking Cancellation & Revenue Intelligence | "
    "Data-driven analysis and baseline machine learning predictions"
)
