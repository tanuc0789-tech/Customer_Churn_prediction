# Customer_Churn_prediction
Customer churn prediction pipeline using Logistic Regression, Decision Trees, and Neural Networks with SMOTE for class imbalance
# Customer Churn Prediction

A machine learning pipeline that predicts whether a customer is likely to
leave (churn) a subscription-based service, using the IBM Telco Customer
Churn dataset.

## 🎯 Goal
Identify customers likely to leave a service or subscription based on
historical usage, billing, and demographic data — so a business can
target them with retention efforts before they leave.

## 📊 Dataset
[Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
(7,043 customers, 21 features) — includes tenure, contract type, monthly
charges, internet/phone service details, and churn status.

## ⚙️ Pipeline
1. **Data Cleaning** — fixed inconsistent `TotalCharges` field, dropped
   identifier column, collapsed redundant categories
2. **Preprocessing** — one-hot encoded categorical variables, scaled
   numeric features
3. **Train/Test Split** — 80/20 stratified split (done *before* balancing
   to avoid data leakage)
4. **Class Imbalance Handling** — applied **SMOTE** to balance the
   minority (churn) class in the training set
5. **Modeling** — trained and compared three models:
   - Logistic Regression
   - Decision Tree
   - Neural Network (MLP)
6. **Evaluation** — precision, recall, F1-score, ROC-AUC, and confusion
   matrix for each model

## 📈 Results

| Model | ROC-AUC |
|---|---|
| **Logistic Regression** | **0.84** |
| Decision Tree | 0.82 |
| Neural Network (MLP) | 0.78 |

Logistic Regression performed best, likely because churn drivers in this
dataset (tenure, contract type, charges) have a fairly linear relationship
with the target.

## 🛠️ Tech Stack
- Python
- pandas, NumPy
- scikit-learn
- imbalanced-learn (SMOTE)

## ▶️ How to Run
```bash
pip install pandas numpy scikit-learn imbalanced-learn
python customer_churn_prediction.py
```
Make sure `WA_Fn-UseC_-Telco-Customer-Churn.csv` is in the same folder as
the script.

## 📁 Output
Saves the best-performing model as `churn_model.pkl` for future predictions.

## ✍️ Author
Tanu Chauhan
