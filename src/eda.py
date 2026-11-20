import matplotlib.pyplot as plt
import seaborn as sns
from config import FIGURES

def create_eda(df):
    out = FIGURES / "eda"
    sns.set_theme(style="whitegrid")
    charts = [
        ("cancellation_distribution.png", lambda ax: sns.countplot(data=df, x="is_canceled", ax=ax),
         "Booking cancellation distribution"),
        ("cancellation_by_hotel.png", lambda ax: (df.groupby("hotel").is_canceled.mean()*100).plot(kind="bar", ax=ax),
         "Cancellation rate by hotel"),
        ("lead_time_by_outcome.png", lambda ax: sns.boxplot(data=df, x="is_canceled", y="lead_time", showfliers=False, ax=ax),
         "Lead time by cancellation outcome"),
        ("adr_distribution.png", lambda ax: sns.histplot(data=df, x="adr", hue="hotel", bins=50, ax=ax),
         "Average daily rate distribution")
    ]
    for filename, draw, title in charts:
        fig, ax = plt.subplots(figsize=(8,5))
        draw(ax); ax.set_title(title); fig.tight_layout()
        fig.savefig(out / filename, dpi=160); plt.close(fig)
    months = ["January","February","March","April","May","June","July","August","September","October","November","December"]
    values = (df.groupby("arrival_date_month").is_canceled.mean()*100).reindex(months)
    fig, ax = plt.subplots(figsize=(10,5))
    values.plot(kind="bar", ax=ax); ax.set_title("Cancellation rate by arrival month")
    ax.set_ylabel("Cancellation rate (%)"); fig.tight_layout()
    fig.savefig(out/"cancellation_by_month.png", dpi=160); plt.close(fig)
