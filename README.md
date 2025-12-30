# Customer Conversion Predictor 📖

## Project Overview
This project tackles a classic business problem: predicting customer conversions to optimize marketing and sales efficiency. Using a real-world dataset from a Portuguese banking institution, we developed a machine learning pipeline to predict whether a client will subscribe to a term deposit. The goal is to shift from a broad, scatter-shot approach to a targeted, data-driven strategy, saving resources and maximizing conversions.

**Dataset:** UCI Machine Learning Repository – [Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/bank+marketing)

---

## 🎯 Business Objective
High costs of ineffective outreach are common in customer acquisition. This project builds an intelligent system to identify clients most likely to convert, allowing teams to focus on high-probability leads and improve ROI.

---

## 🛠️ Technical Approach
- **Data Preprocessing & EDA:** Handle unknown values, encode categorical variables, perform correlation analysis, and create features like call duration bins.  
- **Feature Engineering:** Scale numerical features, one-hot encode categorical features using a `ColumnTransformer`, and combine features like marital/education to capture demographic effects.  
- **Model Benchmarking & Hyperparameter Tuning:** Evaluated 9 classification algorithms and tuned the best-performing XGBoost model using GridSearchCV with 3-fold stratified cross-validation.  
- **Model Evaluation:** Metrics include Accuracy, Precision, Recall, F1-Score, and ROC-AUC. Selected model prioritizes recall to capture most potential conversions.

---

## 🏆 Results & Key Findings
- **Best Model:** XGBoost  
  - Accuracy: 82.8%  
  - ROC-AUC: 90.4%  
  - Recall: 83.7%  

**Insights:**
- Call duration is the strongest predictor; calls >485 seconds led to a 78% conversion rate.  
- Client behavior and engagement matter more than demographics alone.  
- Enables targeted lead scoring for efficient customer conversion.

---

## 💡 Recommendations
- **Deploy Predictive Model:** Score clients before outreach to prioritize high-probability leads.  
- **Quality Over Quantity:** Train agents to engage longer with clients rather than maximize call volume.  
- **Contact Policy:** Limit repeated contacts per client to improve efficiency.

---

## 📂 Repository
**GitHub Repo:** [Customer Conversion Predictor](https://github.com/shervin31/Customer-Conversion-Predictor)

---

## ⚙️ Usage: Load and Use the Pretrained Model
We saved the XGBoost model and preprocessing pipeline in one file for easy deployment.

### 1. Load the Pipeline
```python
import joblib

# Load the saved pipeline
pipeline = joblib.load("bank_marketing_xgboost_pipeline.joblib")
```

### 2. Make Predictions 
```python
# `new_data` must be a pandas DataFrame with the same feature columns as training data
predictions = pipeline.predict(new_data)
predicted_probabilities = pipeline.predict_proba(new_data)[:, 1]  # probability of conversion
```
