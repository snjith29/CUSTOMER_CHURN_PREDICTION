"""Leakage-safe model training and evaluation."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parent
DATA_CANDIDATES = [ROOT / "Churn_Modelling.csv", ROOT / "Churn_Modelling (1).csv", ROOT / "data" / "Churn_Modelling.csv"]
TARGET = "Exited"
NUMERIC = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"]
CATEGORICAL = ["Geography", "Gender"]
DROP = ["RowNumber", "CustomerId", "Surname"]

def load_data():
    path = next((p for p in DATA_CANDIDATES if p.exists()), None)
    if path is None:
        raise FileNotFoundError("Place Churn_Modelling.csv or Churn_Modelling (1).csv in the project root.")
    df = pd.read_csv(path)
    required = set(NUMERIC + CATEGORICAL + [TARGET])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing dataset columns: {sorted(missing)}")
    df = df.drop(columns=[c for c in DROP if c in df.columns]).dropna(subset=[TARGET])
    return df.drop(columns=TARGET), df[TARGET].astype(int), path.name

def build_pipeline(estimator):
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    prep = ColumnTransformer([("num", numeric, NUMERIC), ("cat", categorical, CATEGORICAL)])
    return Pipeline([("preprocess", prep), ("model", estimator)])

def choose_f2_threshold(y, probs):
    thresholds = np.linspace(0.10, 0.90, 81)
    scores = []
    for t in thresholds:
        p = (probs >= t).astype(int)
        precision = precision_score(y, p, zero_division=0)
        recall = recall_score(y, p, zero_division=0)
        scores.append(5 * precision * recall / (4 * precision + recall) if (4 * precision + recall) else 0)
    return float(thresholds[int(np.argmax(scores))])

def main():
    X, y, dataset = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=400, min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=-1),
        "Hist Gradient Boosting": HistGradientBoostingClassifier(max_iter=250, learning_rate=0.08, l2_regularization=1.0, random_state=42),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    trained = {}
    comparison = []
    for name, estimator in candidates.items():
        pipe = build_pipeline(estimator)
        oof = cross_val_predict(pipe, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
        threshold = choose_f2_threshold(y_train, oof)
        pr_auc = average_precision_score(y_train, oof)
        trained[name] = (pipe, threshold, pr_auc)
        comparison.append({"model": name, "cv_pr_auc": round(pr_auc, 4), "cv_roc_auc": round(roc_auc_score(y_train, oof), 4), "threshold": round(threshold, 3), "cv_recall": round(recall_score(y_train, (oof >= threshold).astype(int)), 4)})
    best_name = max(trained, key=lambda name: trained[name][2])
    best_pipe, threshold, _ = trained[best_name]
    best_pipe.fit(X_train, y_train)
    probs = best_pipe.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)
    metrics = {
        "selected_model": best_name, "decision_threshold": threshold, "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0), "recall": recall_score(y_test, preds, zero_division=0),
        "f1": f1_score(y_test, preds, zero_division=0), "roc_auc": roc_auc_score(y_test, probs),
        "pr_auc": average_precision_score(y_test, probs), "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
        "classification_report": classification_report(y_test, preds, target_names=["Stayed", "Churned"], zero_division=0, output_dict=True),
        "dataset": dataset, "test_rows": int(len(y_test)), "churn_rate_test": float(y_test.mean()), "model_comparison": comparison
    }
    joblib.dump({"pipeline": best_pipe, "threshold": threshold, "model_name": best_name}, ROOT / "churn_model.joblib")
    (ROOT / "model_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: v for k, v in metrics.items() if k != "classification_report"}, indent=2))
    print(classification_report(y_test, preds, target_names=["Stayed", "Churned"]))

if __name__ == "__main__":
    main()
