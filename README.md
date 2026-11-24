# Customer Churn Prediction

An end-to-end machine-learning project that estimates which bank customers may leave, compares classification models, and provides an interactive Streamlit dashboard for individual and batch predictions.

## Project highlights

- Leakage-safe preprocessing with Scikit-learn Pipelines.
- Stratified train/test split and five-fold cross-validation.
- Compares Logistic Regression, Random Forest, and HistGradientBoosting.
- Selects the candidate using out-of-fold Precision-Recall AUC.
- Tunes a decision threshold using F2-score to give recall more weight than precision.
- Reports ROC-AUC and PR-AUC from predicted probabilities.
- Saves a reusable model artifact and evaluation metrics.
- Streamlit interface for single-customer and CSV batch scoring.

## Dataset

Uses the Bank Customer Churn / Churn Modelling dataset (10,000 records). The target is `Exited` (1 = churned, 0 = retained).

Required input columns:

| Feature | Meaning |
|---|---|
| CreditScore | Customer credit score |
| Geography | Customer geography |
| Gender | Customer gender |
| Age | Customer age |
| Tenure | Years with the bank |
| Balance | Account balance |
| NumOfProducts | Number of bank products |
| HasCrCard | Credit-card flag (0/1) |
| IsActiveMember | Active-member flag (0/1) |
| EstimatedSalary | Estimated salary |
| Exited | Target label (training only) |

Identifier fields (`RowNumber`, `CustomerId`, and `Surname`) are excluded from model training.

The training script automatically checks for `Churn_Modelling.csv` or `Churn_Modelling (1).csv` in the project root, or `data/Churn_Modelling.csv`.

## Repository structure

```text
CUSTOMER_CHURN_PREDICTION/
├── app.py                  # Streamlit prediction dashboard
├── train_model.py          # Training, model selection, evaluation
├── requirements.txt        # Python dependencies
├── solution.ipynb          # Original exploratory notebook
├── model_metrics.json      # Generated evaluation output
└── churn_model.joblib      # Generated trained pipeline
```

The last two files are generated locally by the training script and need not be committed if you prefer to keep model artifacts out of Git.

## Run locally

### 1. Clone the repository

```bash
git clone https://github.com/snjith29/CUSTOMER_CHURN_PREDICTION.git
cd CUSTOMER_CHURN_PREDICTION
```

### 2. Create an environment and install dependencies

```bash
python -m venv .venv
# Windows:
.venv\\Scripts\\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Add the dataset

Place the CSV file in the project root. Keep either its original name `Churn_Modelling (1).csv` or rename it to `Churn_Modelling.csv`.

### 4. Train and evaluate

```bash
python train_model.py
```

This creates:
- `churn_model.joblib` — preprocessing pipeline and selected estimator.
- `model_metrics.json` — held-out test metrics and model comparison.

### 5. Launch the dashboard

```bash
streamlit run app.py
```

Use the **Single customer** tab to enter customer details, or **Batch CSV scoring** to upload a file and download churn probabilities.

## Methodology and evaluation

1. Split the labelled dataset into stratified 80% training and 20% test partitions.
2. Fit imputation, scaling, and one-hot encoding inside each cross-validation fold. This prevents test-fold information from leaking into preprocessing.
3. Generate out-of-fold training probabilities for each candidate model.
4. Choose the model with the best training PR-AUC, a useful metric when churn and non-churn classes are imbalanced.
5. Choose a decision threshold on training-only out-of-fold predictions using F2-score.
6. Refit the selected pipeline on the complete training partition and evaluate once on the untouched test partition.

The selected threshold is a configurable trade-off: lowering it usually catches more potential churners but can increase false alarms. A business should choose the final operating point based on the cost of missed churn versus retention outreach.

## Important interpretation notes

- Accuracy alone can hide poor churn detection. Review churn precision, recall, F1, PR-AUC, ROC-AUC, and the confusion matrix together.
- A high predicted probability is a risk signal, not proof that a customer will leave.
- This dataset is a learning benchmark. Validate data quality, fairness, and business impact before using predictions for real customer decisions.
- The original `solution.ipynb` is retained as the exploratory notebook; use `train_model.py` for the reproducible pipeline and `app.py` for the interactive demo.

## Future enhancements

- Add SHAP explanations and global/local feature importance.
- Add calibration curves and probability calibration.
- Track experiments and model versions.
- Add automated tests and a CI workflow.
- Deploy the Streamlit app and monitor prediction drift.

## Author

**Sanjith** — [GitHub @snjith29](https://github.com/snjith29)
