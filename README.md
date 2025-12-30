# Customer Conversion Prediction: Predictive Analytics for Targeted Marketing

## 📖 Project Overview
This project tackles a classic business problem: optimizing marketing campaign efficiency. Using a real-world dataset, we developed a machine learning pipeline to predict whether a client will convert. The goal is to shift from broad, scatter-shot marketing to a targeted, data-driven strategy, saving resources and maximizing conversions.

## 🎯 Business Objective
High-cost ineffective marketing is a common challenge. Blindly contacting thousands of clients wastes time and money. This project builds an intelligent system to identify clients most likely to convert, allowing marketing efforts to focus where they are most effective.

## 📊 Dataset
The dataset contains information on over 11,000 client contacts.

- **Features**: 16 attributes including demographics (age, job, education), previous campaign contacts, and macroeconomic indicators.  
- **Target Variable**: A binary field indicating whether a client subscribed (`yes`/`no`).

## 🛠️ Technical Approach
The project follows the CRISP-DM data science lifecycle:

### Data Preprocessing & EDA
- Handled unknown values and encoded categorical variables.  
- Performed correlation analysis and feature engineering (e.g., duration bins).  
- Used stratified sampling to maintain class balance in train/test splits.

### Feature Engineering
- Applied `StandardScaler` to numerical and `OneHotEncoder` to categorical features via `ColumnTransformer`.  
- Created new features combining demographic attributes (e.g., marital + education).

### Model Benchmarking & Hyperparameter Tuning
- Benchmarked 9 classification algorithms (Logistic Regression, Random Forest, XGBoost, etc.) on speed and accuracy.  
- Used `GridSearchCV` with 3-fold Stratified Cross-Validation to optimize hyperparameters and prevent overfitting.

### Model Evaluation
- Evaluated models with Accuracy, Precision, Recall, F1-Score, and ROC-AUC.  
- Final model chosen based on generalization and high recall to capture most potential converters.

## 🏆 Results & Key Findings
**XGBoost** emerged as the best-performing model:  
- **Accuracy:** 82.8%  
- **ROC-AUC:** 90.4%  
- **Recall:** 83.7%  

### 🔍 Critical Business Insights
- **Call Duration is Key:** Calls longer than 485 seconds had a 78% conversion rate.  
- **Model-Driven Targeting:** The model accurately scores clients, enabling prioritization of high-probability leads.  
- **Beyond Demographics:** Client behavior and engagement are stronger predictors than demographics alone.

## 💡 Data-Driven Recommendations
- **Deploy the Predictive Model:** Integrate XGBoost into lead management to prioritize high-probability clients.  
- **Quality Over Quantity:** Train agents to focus on engagement and longer conversations rather than volume.  
- **Implement a Call Policy:** Limit repeated contacts per client to improve overall efficiency.
