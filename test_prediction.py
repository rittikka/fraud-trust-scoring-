
import pandas as pd
import joblib

# Load dataset
df = pd.read_csv("data/anchal_seller_temporal_features_final.csv")

# Load saved model and imputer
model = joblib.load("models/seller_risk_lightgbm.pkl")
imputer = joblib.load("models/seller_risk_imputer.pkl")

# Select the 30 numerical features in their original order
X = df.drop(columns=["seller_id", "date"])

# Use only the first 5 rows for this test
X_test = X.head(5)

# Handle missing values using the saved imputer
X_test_imputed = imputer.transform(X_test)

# Predict behavioural risk probability
risk_probabilities = model.predict_proba(X_test_imputed)[:, 1]

# Convert risk probability into provisional trust score
results = df.loc[X_test.index, ["seller_id", "date"]].copy()
results["risk_probability"] = risk_probabilities
results["provisional_trust_score"] = (
    100 * (1 - results["risk_probability"])
).round(2)

print("\nPrediction test results:\n")
print(results.to_string(index=False))
