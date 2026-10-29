
from pathlib import Path
import pandas as pd


def calculate_trust_score(risk_probability):
    probability = float(risk_probability)

    if not 0 <= probability <= 1:
        raise ValueError("Risk probability must be between 0 and 1.")

    return round(100 * (1 - probability), 2)


def calculate_behavioral_trust_score(features):
    weights = {
        "delivery_latency_deviation_abs": 0.35,
        "review_sentiment_slope": 0.30,
        "order_value_velocity_pct": 0.35,
    }

    penalties = {}

    delivery = features.get("delivery_latency_deviation_abs")
    if pd.notna(delivery):
        penalties["delivery_latency_deviation_abs"] = min(
            abs(float(delivery)) / 30, 1
        )

    review = features.get("review_sentiment_slope")
    if pd.notna(review):
        penalties["review_sentiment_slope"] = min(
            max(-float(review) / 4, 0), 1
        )

    sales = features.get("order_value_velocity_pct")
    if pd.notna(sales):
        penalties["order_value_velocity_pct"] = min(
            abs(float(sales)) / 100, 1
        )

    if not penalties:
        return None

    total_weight = sum(weights[key] for key in penalties)

    risk = sum(
        weights[key] * penalties[key]
        for key in penalties
    ) / total_weight

    return round(100 * (1 - risk), 2)


if __name__ == "__main__":
    risk_probability = 0.20
    score = calculate_trust_score(risk_probability)

    print("Behavioural Risk Probability:", risk_probability)
    print("Provisional Trust Score:", score, "/ 100")
