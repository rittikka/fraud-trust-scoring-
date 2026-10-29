
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

input_path = BASE_DIR / "data" / "processed" / "seller_trust_scores.csv"
output_path = BASE_DIR / "data" / "processed" / "seller_level_trust_scores.csv"

df = pd.read_csv(input_path)

seller_summary = (
    df.groupby("seller_id", as_index=False)
    .agg(
        average_trust_score=("behavioral_trust_score", "mean"),
        minimum_trust_score=("behavioral_trust_score", "min"),
        maximum_trust_score=("behavioral_trust_score", "max"),
        days_scored=("behavioral_trust_score", "count"),
    )
)

seller_summary["average_trust_score"] = (
    seller_summary["average_trust_score"].round(2)
)

seller_summary = seller_summary.sort_values(
    "average_trust_score", na_position="last"
)

seller_summary.to_csv(output_path, index=False)

print("Seller-level summary generated.")
print("Total sellers:", len(seller_summary))
print("Sellers with scores:", seller_summary["average_trust_score"].notna().sum())
print("Saved to:", output_path)
print(seller_summary.head(10).to_string(index=False))
