# Customer Churn Prediction

A machine learning classification project that predicts whether a bank customer is likely to leave the bank, using customer demographics and account-related information.

## Project Overview

Customer churn (also called customer attrition) occurs when a customer stops using a company's products or services. Identifying customers who may leave helps businesses understand retention risks and plan targeted customer engagement.

This project explores the **Churn Modelling** dataset and builds a **Logistic Regression** model to classify customers as:
- **0 — Retained:** the customer did not leave.
- **1 — Churned:** the customer left the bank.

## Dataset

The notebook uses the Churn Modelling dataset containing **10,000 customer records** and the following fields:

| Feature | Description |
|---|---|
| CreditScore | Customer's credit score |
| Geography | Customer's country/region |
| Gender | Customer's gender |
| Age | Customer's age |
| Tenure | Number of years with the bank |
| Balance | Account balance |
| NumOfProducts | Number of bank products held |
| HasCrCard | Whether the customer has a credit card |
| IsActiveMember | Whether the customer is an active member |
| EstimatedSalary | Estimated customer salary |
| Exited | Target: 1 if the customer exited, otherwise 0 |

The notebook removes the identifier/name columns `RowNumber`, `CustomerId`, and `Surname` before model training.

> **Dataset filename:** The notebook reads `Churn_Modelling.csv`. If your downloaded file has a different name (for example, `Churn_Modelling (1).csv`), rename it to `Churn_Modelling.csv` before running the notebook.

## Tech Stack

- Python
- Jupyter Notebook / Google Colab
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn

## Workflow

1. **Load the data** — read the CSV dataset and inspect its structure and summary statistics.
2. **Preprocess the data** — remove identifier columns and check for missing values.
3. **Encode categorical features** — apply one-hot encoding to `Geography` and `Gender`.
4. **Explore the target** — visualize the distribution of customers who stayed and exited.
5. **Scale features** — apply `StandardScaler` to the input features.
6. **Split the data** — create training and testing sets using a 70:30 split and `random_state=42`.
7. **Train the model** — fit a Logistic Regression classifier.
8. **Evaluate** — review accuracy, the classification report, F1-score, recall, ROC-AUC score, and confusion matrix.

## Model and Results

The notebook's recorded test-set results are:

| Metric | Result |
|---|---:|
| Accuracy | **81%** |
| Churn precision (class 1) | 54% |
| Churn recall (class 1) | 20% |
| Churn F1-score (class 1) | 29% |
| ROC-AUC score (as calculated in notebook) | 0.579 |

**Confusion matrix**

| | Predicted: Stayed | Predicted: Churned |
|---|---:|---:|
| Actual: Stayed | 2,318 | 98 |
| Actual: Churned | 468 | 116 |

The model identifies a portion of customers who churn, but its **20% churn recall** shows that many churn cases are missed. Accuracy alone is therefore not enough to judge this model; recall, F1-score, and the confusion matrix are especially important for a churn use case.

*These figures are the outputs saved in the notebook and may vary if the data or training process changes. The notebook calculates ROC-AUC using predicted class labels; for a more informative ROC-AUC evaluation, use the model's churn probabilities (for example, `model.predict_proba(X_test)[:, 1]`).*

## Repository Structure

```text
CUSTOMER_CHURN_PREDICTION/
├── Churn_Modelling (1).csv   # Dataset (rename as noted above if needed)
├── solution.ipynb            # Data analysis, training, and evaluation
├── feature_importance.png    # Visualization asset, if included in the repository
└── README.md
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/snjith29/CUSTOMER_CHURN_PREDICTION.git
cd CUSTOMER_CHURN_PREDICTION
```

### 2. Install dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

### 3. Prepare the dataset

Make sure the CSV file is named `Churn_Modelling.csv` and is in the same directory as the notebook. The notebook also contains a Google Colab upload step if the file is not found.

### 4. Run the notebook

```bash
jupyter notebook solution.ipynb
```

Run the notebook cells in order. You can also upload the notebook and dataset to Google Colab.

## Possible Improvements

- Split the data before fitting the scaler, and use a Scikit-learn Pipeline to prevent data leakage.
- Use stratified splitting and address class imbalance with class weights or resampling.
- Tune the classification threshold to improve churn recall.
- Compare Logistic Regression with Random Forest, Gradient Boosting, and other classifiers.
- Evaluate with cross-validation, ROC curves, and precision-recall curves.
- Save the trained preprocessing and model pipeline for repeatable predictions.
- Build a Streamlit or Flask interface for interactive churn predictions.

## Author

**Sanjith**  
GitHub: [@snjith29](https://github.com/snjith29)

---

If you find this project useful, consider giving the repository a ⭐.
