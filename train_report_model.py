
from pathlib import Path

import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
)

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "processed" / "seller_report_features_labeled.csv"
OUT = BASE / "models" / "report_model"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["date", "seller_id"]).reset_index(drop=True)

# Exclude identifiers and target from model inputs
drop_cols = ["seller_id", "date", "risk_label"]
feature_cols = [c for c in df.columns if c not in drop_cols]

split_date = pd.Timestamp("2018-05-03")
train = df[df["date"] < split_date].copy()
test = df[df["date"] >= split_date].copy()

X_train = train[feature_cols].apply(pd.to_numeric, errors="coerce")
X_test = test[feature_cols].apply(pd.to_numeric, errors="coerce")
y_train = train["risk_label"].astype(int)
y_test = test["risk_label"].astype(int)

# Fit imputer only on training data
imputer = SimpleImputer(strategy="median")
X_train_imp = imputer.fit_transform(X_train)
X_test_imp = imputer.transform(X_test)

model = LGBMClassifier(
    objective="binary",
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
    verbosity=-1,
)

model.fit(X_train_imp, y_train)
prob = model.predict_proba(X_test_imp)[:, 1]
pred = (prob >= 0.5).astype(int)

print("Rows:", len(df))
print("Features:", len(feature_cols))
print("Split date:", split_date.date())
print("Train rows:", len(train), "| Test rows:", len(test))
print("\nEvaluation")
print("Accuracy:", round(accuracy_score(y_test, pred), 4))
print("Precision:", round(precision_score(y_test, pred, zero_division=0), 4))
print("Recall:", round(recall_score(y_test, pred, zero_division=0), 4))
print("F1:", round(f1_score(y_test, pred, zero_division=0), 4))
print("ROC-AUC:", round(roc_auc_score(y_test, prob), 4))
print("PR-AUC:", round(average_precision_score(y_test, prob), 4))
print("Positive-class baseline:", round(y_test.mean(), 4))
print("Confusion matrix:\n", confusion_matrix(y_test, pred))

# Save separately so the current app model is not overwritten
joblib.dump(model, OUT / "seller_risk_lightgbm.pkl")
joblib.dump(imputer, OUT / "seller_risk_imputer.pkl")
joblib.dump(feature_cols, OUT / "feature_columns.pkl")

predictions = test[["seller_id", "date", "risk_label"]].copy()
predictions["predicted_risk_probability"] = prob
predictions["predicted_risk_label"] = pred
predictions["provisional_trust_score"] = 100 * (1 - prob)
predictions.to_csv(OUT / "seller_risk_predictions.csv", index=False)

print("\nSaved new model artifacts to:", OUT)
print("Existing model and dashboard files were not changed.")
