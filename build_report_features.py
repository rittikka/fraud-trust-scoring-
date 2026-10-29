
from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "olist_raw"
OUTPUT_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

orders = pd.read_csv(RAW_DIR / "olist_orders_dataset.csv")
items = pd.read_csv(RAW_DIR / "olist_order_items_dataset.csv")
reviews = pd.read_csv(RAW_DIR / "olist_order_reviews_dataset.csv")

reviews["review_comment_message"] = (
    reviews["review_comment_message"].fillna("").astype(str)
)

orders["order_purchase_timestamp"] = pd.to_datetime(
    orders["order_purchase_timestamp"], errors="coerce"
)
orders["order_delivered_customer_date"] = pd.to_datetime(
    orders["order_delivered_customer_date"], errors="coerce"
)
orders["order_estimated_delivery_date"] = pd.to_datetime(
    orders["order_estimated_delivery_date"], errors="coerce"
)
reviews["review_creation_date"] = pd.to_datetime(
    reviews["review_creation_date"], errors="coerce"
)

orders["delivery_delay_days"] = (
    orders["order_delivered_customer_date"]
    - orders["order_estimated_delivery_date"]
).dt.total_seconds() / 86400

items = items.dropna(subset=["seller_id", "order_id"]).copy()
items["price"] = pd.to_numeric(items["price"], errors="coerce")

seller_orders = items.merge(
    orders[
        [
            "order_id",
            "order_purchase_timestamp",
            "delivery_delay_days",
            "order_status",
        ]
    ],
    on="order_id",
    how="inner",
)

seller_orders = seller_orders.dropna(
    subset=["order_purchase_timestamp"]
)
seller_orders = seller_orders[
    seller_orders["order_status"] == "delivered"
].copy()

seller_orders["date"] = (
    seller_orders["order_purchase_timestamp"].dt.normalize()
)

daily = (
    seller_orders.groupby(["seller_id", "date"], as_index=False)
    .agg(
        daily_order_count=("order_id", "nunique"),
        daily_sales_value=("price", "sum"),
        average_delivery_delay=("delivery_delay_days", "mean"),
        late_delivery_rate=(
            "delivery_delay_days",
            lambda x: (x > 0).mean(),
        ),
    )
)

daily = daily.sort_values(
    ["seller_id", "date"]
).reset_index(drop=True)

for window in (7, 30, 90):
    for col in (
        "daily_order_count",
        "daily_sales_value",
        "average_delivery_delay",
        "late_delivery_rate",
    ):
        daily[f"{col}_rolling_{window}d"] = (
            daily.groupby("seller_id")[col]
            .transform(
                lambda s: s.rolling(window, min_periods=1).mean()
            )
        )

daily["seller_historical_median_delay"] = (
    daily.groupby("seller_id")["average_delivery_delay"]
    .transform(
        lambda s: s.shift(1).expanding(min_periods=1).median()
    )
)

daily["delivery_latency_deviation"] = (
    daily["average_delivery_delay"]
    - daily["seller_historical_median_delay"]
)

daily["delivery_latency_deviation_abs"] = (
    daily["delivery_latency_deviation"].abs()
)

daily["previous_sales_value"] = (
    daily.groupby("seller_id")["daily_sales_value"].shift(1)
)

daily["order_value_velocity"] = (
    daily["daily_sales_value"] - daily["previous_sales_value"]
)

daily["order_value_velocity_pct"] = (
    (
        daily["daily_sales_value"] - daily["previous_sales_value"]
    )
    / daily["previous_sales_value"].replace(0, np.nan)
) * 100

review_orders = (
    reviews[["order_id", "review_score"]]
    .dropna(subset=["review_score"])
    .merge(
        items[["order_id", "seller_id"]].drop_duplicates(),
        on="order_id",
        how="inner",
    )
    .merge(
        orders[["order_id", "order_purchase_timestamp"]],
        on="order_id",
        how="inner",
    )
)

review_orders["date"] = (
    review_orders["order_purchase_timestamp"].dt.normalize()
)

review_daily = (
    review_orders.groupby(["seller_id", "date"], as_index=False)
    .agg(
        average_review_score=("review_score", "mean"),
        review_count=("review_score", "count"),
    )
)

daily = daily.merge(
    review_daily, on=["seller_id", "date"], how="left"
)

daily = daily.sort_values(
    ["seller_id", "date"]
).reset_index(drop=True)

daily["previous_review_score"] = (
    daily.groupby("seller_id")["average_review_score"].shift(1)
)

daily["review_score_change"] = (
    daily["average_review_score"] - daily["previous_review_score"]
)

daily["review_sentiment_proxy"] = daily["average_review_score"]

daily["review_sentiment_slope"] = (
    daily.groupby("seller_id")["review_sentiment_proxy"]
    .transform(lambda s: s - s.shift(1))
)

output_path = OUTPUT_DIR / "seller_report_features.csv"
daily.to_csv(output_path, index=False)

model_ready = daily.copy()
clean_output_path = (
    OUTPUT_DIR / "seller_report_features_model_ready.csv"
)
model_ready.to_csv(clean_output_path, index=False)

final_features = daily[
    [
        "seller_id",
        "date",
        "delivery_latency_deviation_abs",
        "review_sentiment_slope",
        "order_value_velocity_pct",
    ]
].copy()

final_path = OUTPUT_DIR / "seller_five_core_features.csv"
final_features.to_csv(final_path, index=False)

print("Feature engineering completed.")
print("Rows:", len(daily))
print("Total generated columns:", len(daily.columns))
print("Full features saved to:", output_path)
print("Model-ready copy saved to:", clean_output_path)
print("Five core features saved to:", final_path)
print("Available behavioural parameters: 3 out of 5")
print("Refund rate and dispute ratio require additional data.")
