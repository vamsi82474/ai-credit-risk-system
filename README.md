# 💳 AI Credit Risk & Collections Decision System

## 📌 Overview
End-to-end machine learning project to predict customer delinquency risk and support collections strategy.  
Includes model training, prediction pipeline, Streamlit deployment, explainability, and business recommendations.

---

## 🚀 Features
- Predicts delinquency risk using machine learning
- Real-time scoring via Streamlit app
- Clear justification (“why this prediction?”)
- Business recommendations for collections teams
- Handles class imbalance and improves risk detection

---

## 🧠 Model
- Algorithm: Random Forest (class_weight="balanced")
- Key features:
  - Credit Utilization
  - Missed Payments
  - Income
  - Age
- Includes threshold tuning + rule-based override for high-risk cases

---

## 📊 Workflow
1. Data Cleaning & Preprocessing  
2. Feature Encoding  
3. Model Training  
4. Prediction System  
5. Streamlit Deployment  

---

## ⚠️ Key Insight
The initial model showed high accuracy but failed to detect delinquent customers due to class imbalance.  
This was fixed using:
- Balanced training  
- Threshold tuning (0.35 instead of 0.5)  
- Rule-based override for extreme high-risk scenarios  

---

## 💼 Business Impact
- Identifies high-risk customers early  
- Enables proactive collections strategy  
- Improves recovery efficiency  
- Supports data-driven decisions  

---

## ⚖️ Responsible AI
- Bias-aware modeling  
- Explainable predictions  
- Human-in-the-loop decision support  
- Uses financial behavior only (no sensitive attributes)  

---

## 🖥️ Run Locally

```bash
pip install -r requirements.txt
python src/train.py
streamlit run app/app.py

nshot 2026-05-04 172935" src="https://github.com/user-attachments/assets/ee3b943d-cd4f-4cc2-af6d-152e1dfead82" />
