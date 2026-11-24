"""Interactive Streamlit dashboard for churn-risk scoring."""
from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "churn_model.joblib"
METRICS_PATH = ROOT / "model_metrics.json"
FEATURES = ["CreditScore", "Geography", "Gender", "Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"]

st.set_page_config(page_title="Customer Churn Intelligence", page_icon="📊", layout="wide")
st.title("Customer Churn Intelligence")
st.caption("Estimate churn risk and prioritize customer retention reviews.")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None

artifact = load_model()
if artifact is None:
    st.warning("Model file not found. Run the training command first.")
    st.code("pip install -r requirements.txt\npython train_model.py\nstreamlit run app.py")
    st.stop()

pipeline, threshold = artifact["pipeline"], float(artifact["threshold"])
with st.sidebar:
    st.subheader("Model")
    st.write(artifact["model_name"])
    st.metric("Decision threshold", f"{threshold:.2f}")
    st.caption("Threshold selected using training-only out-of-fold predictions.")

single, batch, info = st.tabs(["Single customer", "Batch CSV scoring", "Model methodology"])
with single:
    st.subheader("Customer details")
    with st.form("customer_form"):
        a, b, c = st.columns(3)
        with a:
            credit = st.number_input("Credit score", 300, 900, 650)
            geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
            gender = st.selectbox("Gender", ["Female", "Male"])
            age = st.number_input("Age", 18, 100, 40)
        with b:
            tenure = st.number_input("Tenure (years)", 0, 10, 5)
            balance = st.number_input("Balance", min_value=0.0, value=75000.0, step=1000.0)
            products = st.selectbox("Number of products", [1, 2, 3, 4], index=1)
        with c:
            card = st.selectbox("Has credit card?", ["Yes", "No"])
            active = st.selectbox("Active member?", ["Yes", "No"])
            salary = st.number_input("Estimated salary", min_value=0.0, value=80000.0, step=1000.0)
        submitted = st.form_submit_button("Estimate churn risk", type="primary")
    if submitted:
        customer = pd.DataFrame([{"CreditScore": credit, "Geography": geography, "Gender": gender, "Age": age, "Tenure": tenure, "Balance": balance, "NumOfProducts": products, "HasCrCard": int(card == "Yes"), "IsActiveMember": int(active == "Yes"), "EstimatedSalary": salary}])
        probability = float(pipeline.predict_proba(customer)[0, 1])
        high_risk = probability >= threshold
        left, right = st.columns(2)
        left.metric("Churn probability", f"{probability:.1%}")
        right.metric("Risk category", "Higher risk" if high_risk else "Lower risk")
        st.progress(min(max(probability, 0.0), 1.0))
        if high_risk:
            st.warning("Above the selected threshold. Consider a retention review.")
        else:
            st.success("Below the selected threshold.")
        st.caption("This is a model estimate, not a guarantee of customer behaviour.")

with batch:
    st.subheader("Score multiple customers")
    st.write("Upload a CSV containing the ten model input columns. Identifier columns and Exited are optional.")
    file = st.file_uploader("Customer CSV", type=["csv"])
    if file:
        df = pd.read_csv(file).drop(columns=["Exited", "RowNumber", "CustomerId", "Surname"], errors="ignore")
        missing = sorted(set(FEATURES) - set(df.columns))
        if missing:
            st.error("Missing columns: " + ", ".join(missing))
        else:
            scored = df.copy()
            scored["churn_probability"] = pipeline.predict_proba(df[FEATURES])[:, 1]
            scored["risk_level"] = scored["churn_probability"].apply(lambda p: "Higher" if p >= threshold else "Lower")
            st.dataframe(scored, use_container_width=True)
            st.download_button("Download predictions CSV", scored.to_csv(index=False).encode("utf-8"), "churn_predictions.csv", "text/csv")

with info:
    st.subheader("Methodology")
    st.markdown("""
    - Stratified 80/20 train-test split.
    - Imputation, scaling, and one-hot encoding happen inside a Scikit-learn pipeline, fitted only on training folds.
    - Compares Logistic Regression, Random Forest, and HistGradientBoosting using five-fold out-of-fold probabilities.
    - Selects the model by training PR-AUC and selects a decision threshold using F2-score.
    - Evaluates the final model on a held-out test set; ROC-AUC and PR-AUC use probabilities.
    
    Use this as decision support only. Check data quality, fairness, and business outcomes before acting on predictions.
    """)
    if METRICS_PATH.exists():
        with open(METRICS_PATH, encoding="utf-8") as f:
            metrics = json.load(f)
        st.json(metrics)
