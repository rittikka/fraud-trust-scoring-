
import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Seller Trust Scoring",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Explainable Seller Trust Scoring")
st.caption(
    "Behavioural risk assessment using seller sales, "
    "delivery and review patterns"
)

st.divider()
st.subheader("Experimental Five-Feature Trust Scores")

summary_path = "data/processed/seller_level_trust_scores.csv"

try:
    seller_summary = pd.read_csv(summary_path)
    seller_summary["score_reliability"] = seller_summary["days_scored"].apply(
        lambda days: (
            "Low" if days <= 1
            else "Limited" if days <= 7
            else "Better coverage" if days <= 30
            else "Strong coverage"
        )
    )

    st.write(
        "Seller-level scores calculated from delivery deviation, "
        "review-score trend, and sales-value change."
    )

    st.dataframe(
        seller_summary,
        use_container_width=True
    )

except FileNotFoundError:
    st.warning("Generate seller-level trust scores first.")


@st.cache_data
def load_data():
    return pd.read_csv("data/anchal_seller_temporal_features_final.csv")


@st.cache_resource
def load_model():
    model = joblib.load("models/seller_risk_lightgbm.pkl")
    imputer = joblib.load("models/seller_risk_imputer.pkl")
    return model, imputer


try:
    df = load_data()
    model, imputer = load_model()

    st.success("Dataset and model loaded successfully.")

    sellers = sorted(df["seller_id"].dropna().unique())

    selected_seller = st.selectbox(
        "Select a seller",
        sellers
    )

    seller_data = df[df["seller_id"] == selected_seller].copy()
    seller_data["date"] = pd.to_datetime(seller_data["date"])
    seller_data = seller_data.sort_values("date")

    selected_date = st.selectbox(
        "Select a date",
        seller_data["date"].dt.strftime("%Y-%m-%d").tolist()
    )

    row = seller_data[
        seller_data["date"].dt.strftime("%Y-%m-%d") == selected_date
    ].iloc[[0]]

    features = row.drop(columns=["seller_id", "date"])

    X = df.drop(columns=["seller_id", "date"])
    features = features[X.columns]

    available_features = int(features.notna().sum(axis=1).iloc[0])

    st.write(
        "Available feature values:",
        available_features,
        "out of",
        features.shape[1]
    )

    st.subheader("Seller Behaviour Features")
    st.subheader("Experimental Behavioural Trust Score")
    st.caption(
        "Score uses three behavioural features. Lower penalty "
        "contributes to a higher experimental trust score."
    )

    experimental_path = "data/processed/seller_trust_scores.csv"
    experimental_df = pd.read_csv(experimental_path)
    experimental_df["date"] = pd.to_datetime(experimental_df["date"])

    selected_experimental = experimental_df[
        (experimental_df["seller_id"] == selected_seller)
        & (
            experimental_df["date"].dt.strftime("%Y-%m-%d")
            == selected_date
        )
    ]

    if not selected_experimental.empty:
        score_row = selected_experimental.iloc[0]
        experimental_score = selected_experimental[
            "behavioral_trust_score"
        ].iloc[0]

        if pd.notna(experimental_score):
            st.metric(
                "Experimental Trust Score",
                f"{experimental_score:.2f}/100"
            )
            
            st.subheader("Behavioural Feature Breakdown")

            feature_details = pd.DataFrame({
                "Feature": [
                    "Delivery Latency Deviation",
                    "Review Score Change",
                    "Order-Value Velocity (%)"
                ],
                "Value": [
                    selected_experimental["delivery_latency_deviation_abs"].iloc[0],
                    selected_experimental["review_sentiment_slope"].iloc[0],
                    selected_experimental["order_value_velocity_pct"].iloc[0]
                ]
            })

            st.dataframe(feature_details, use_container_width=True)

        else:
            st.warning("Experimental score unavailable for this date.")
    else:
        st.info("No experimental score found for this seller and date.")

    feature_table = features.T.copy()
    feature_table.columns = ["Value"]
    feature_table.index.name = "Feature"
    st.dataframe(feature_table, use_container_width=True)
    st.subheader("Penalty Breakdown")
    delivery = score_row.get("delivery_latency_deviation_abs")
    review = score_row.get("review_sentiment_slope")
    sales = score_row.get("order_value_velocity_pct")

    penalty_details = pd.DataFrame({
        "Feature": [
                    "Delivery Latency Deviation",
                    "Review Score Change",
                    "Order-Value Velocity",
                ],
                "Penalty": [
                    min(abs(float(delivery)) / 30, 1) if pd.notna(delivery) else None,
                    min(max(-float(review) / 4, 0), 1) if pd.notna(review) else None,
                    min(abs(float(sales)) / 100, 1) if pd.notna(sales) else None,
                ],
                "Weight": [0.35, 0.30, 0.35],
            })

    st.dataframe(penalty_details, use_container_width=True)

    if available_features == 0:
        st.warning(
            "Trust score unavailable: all behavioural features are missing "
            "for this seller and date. Please select another date."
        )
    else:
        X_imputed = imputer.transform(features)
        risk_probability = float(model.predict_proba(X_imputed)[0, 1])
        trust_score = round(100 * (1 - risk_probability), 2)

        st.divider()
        st.subheader("Seller Assessment")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Behavioural Risk Probability",
                f"{risk_probability:.2%}"
            )

        with col2:
            st.metric(
                "Provisional Trust Score",
                f"{trust_score}/100"
            )

        st.caption(
            "This is a provisional behavioural-risk score, not proof of fraud "
            "or a calibrated measure of trustworthiness."
        )

        st.divider()
        st.subheader("Why did the model give this score?")

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_imputed)

        if isinstance(shap_values, list):
            row_shap_values = shap_values[1][0]
        elif len(shap_values.shape) == 3:
            row_shap_values = shap_values[0, :, 1]
        else:
            row_shap_values = shap_values[0]

        explanation = pd.DataFrame({
            "Feature": features.columns,
            "Feature Value": features.iloc[0].values,
            "SHAP Impact": row_shap_values
        })

        explanation["Absolute Impact"] = explanation["SHAP Impact"].abs()
        explanation = explanation.sort_values(
            "Absolute Impact", ascending=False
        ).head(10)

        st.write(
            "These are the 10 features with the largest SHAP impact "
            "for the selected seller-date."
        )

        st.dataframe(
            explanation.drop(columns=["Absolute Impact"]),
            use_container_width=True
        )

except Exception as e:
    st.error(f"Could not load or assess seller: {e}")
