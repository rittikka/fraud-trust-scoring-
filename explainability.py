
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# Load dataset and trained model
df = pd.read_csv("data/anchal_seller_temporal_features_final.csv")
model = joblib.load("models/seller_risk_lightgbm.pkl")
imputer = joblib.load("models/seller_risk_imputer.pkl")

# Select numerical features in the original order
X = df.drop(columns=["seller_id", "date"])

# Explain a small sample first
X_sample = X.head(100)
X_sample_imputed = imputer.transform(X_sample)

# Create SHAP explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample_imputed)

# Handle binary classification output
if isinstance(shap_values, list):
    shap_values = shap_values[1]

# Display feature importance
shap.summary_plot(
    shap_values,
    X_sample,
    show=False
)

plt.tight_layout()
plt.savefig("shap_summary.png", dpi=300, bbox_inches="tight")
plt.show()

print("SHAP summary saved as shap_summary.png")
