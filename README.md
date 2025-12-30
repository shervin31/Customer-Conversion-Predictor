# Customer Conversion Predictor 📖

## Project Overview
This project addresses a common business challenge: predicting customer conversions to optimize marketing efficiency. Using a real-world dataset from a Portuguese banking institution, we developed a machine learning pipeline to predict whether a client will subscribe to a term deposit. The goal is to move from broad outreach to a targeted, data-driven strategy, saving resources and maximizing conversions.

**Dataset:** UCI Machine Learning Repository – Bank Marketing Dataset  

---

## 🎯 Business Objective
Ineffective outreach is costly. This project builds an intelligent system to identify clients most likely to convert, enabling teams to focus on high-probability leads and improve ROI.

---

## 🛠️ Technical Approach
**Data Preprocessing & EDA:**  
- Handle unknown values  
- Encode categorical variables  
- Analyze correlations  
- Create features like call duration bins  

**Feature Engineering:**  
- Scale numerical features  
- One-hot encode categorical features using a `ColumnTransformer`  
- Combine features like marital/education to capture demographic effects  

**Model Benchmarking & Hyperparameter Tuning:**  
- Evaluated 9 classification algorithms  
- Tuned XGBoost using `GridSearchCV` with 3-fold stratified cross-validation  

**Model Evaluation:**  
- Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC  
- Model selection prioritizes **recall** to capture the most potential conversions  

---

## 🏆 Results & Key Findings
**Best Model:** XGBoost  
- Accuracy: 82.8%  
- ROC-AUC: 90.4%  
- Recall: 83.7%  

**Insights:**  
- Call duration is the strongest predictor; calls >485 seconds led to a 78% conversion rate  
- Client behavior and engagement matter more than demographics alone  
- Enables targeted lead scoring for efficient customer conversion  

---

## 💡 Recommendations
- **Deploy Predictive Model:** Score clients before outreach to prioritize high-probability leads  
- **Quality Over Quantity:** Train agents to engage longer with clients rather than maximize call volume  
- **Contact Policy:** Limit repeated contacts per client to improve efficiency  

---

## ⚙️ Deployment & Usage

### 1. Install Dependencies
```bash
pip install pandas numpy scikit-learn xgboost joblib
```

### 2. Load the Pretrained Pipeline 
``` python
import joblib

# Load the saved pipeline (preprocessing + XGBoost model)
pipeline = joblib.load("bank_marketing_xgboost_pipeline.joblib")
```

### 3. Make Predictions 
``` python
# `new_data` must be a pandas DataFrame with the same feature columns as training data
predictions = pipeline.predict(new_data)  # predicted classes (0=no, 1=yes)
predicted_probabilities = pipeline.predict_proba(new_data)[:, 1]  # probability of conversion
```
