import pandas as pd
import joblib
import os

# Load features and labels
features = pd.read_csv("data/anchal_seller_temporal_features_final.csv")
labels = pd.read_csv("data/seller_risk_labels.csv")

features["date"] = pd.to_datetime(features["date"])
labels["date"] = pd.to_datetime(labels["date"])

# Merge using original seller-date keys
df = features.merge(
    labels[["seller_id", "date", "risk_label"]],
    on=["seller_id", "date"],
    how="inner",
    validate="one_to_one"
)

# Recreate the original chronological split
split_date = df["date"].quantile(0.80)
train_end = split_date - pd.Timedelta(days=30)

train_mask = df["date"] < train_end
test_mask = df["date"] >= split_date

X_test = df.loc[test_mask].drop(
    columns=["seller_id", "date", "risk_label"]
)

# Load the already-trained model and imputer
model = joblib.load("models/seller_risk_lightgbm.pkl")
imputer = joblib.load("models/seller_risk_imputer.pkl")

# Ensure correct feature order
X_test = X_test[list(imputer.feature_names_in_)]

# Predict on the original test records
X_test_imputed = imputer.transform(X_test)
probabilities = model.predict_proba(X_test_imputed)[:, 1]
predictions = (probabilities >= 0.5).astype(int)

# Save predictions with true labels
results = df.loc[
    test_mask, ["seller_id", "date", "risk_label"]
].copy()

results["predicted_risk_probability"] = probabilities
results["predicted_risk_label"] = predictions

os.makedirs("data/processed", exist_ok=True)
results.to_csv("data/processed/seller_risk_predictions.csv", index=False)

print("Saved: data/processed/seller_risk_predictions.csv")
print("Rows:", len(results))
print(results.head())