
from pathlib import Path
import pandas as pd
from trust_score import calculate_behavioral_trust_score

BASE_DIR = Path(__file__).resolve().parent

input_path = BASE_DIR / "data" / "processed" / "seller_five_core_features.csv"
output_path = BASE_DIR / "data" / "processed" / "seller_trust_scores.csv"

df = pd.read_csv(input_path)

feature_columns = [
    "delivery_latency_deviation_abs",
    "review_sentiment_slope",
    "order_value_velocity_pct",
]

df["behavioral_trust_score"] = df.apply(
    lambda row: calculate_behavioral_trust_score(row[feature_columns]),
    axis=1,
)

df.to_csv(output_path, index=False)

print("Trust scores generated successfully.")
print("Total records:", len(df))
print("Valid scores:", df["behavioral_trust_score"].notna().sum())
print("Saved to:", output_path)
print(df[["seller_id", "date", "behavioral_trust_score"]].head())
