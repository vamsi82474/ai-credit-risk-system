import streamlit as st
import pandas as pd
import joblib

# Load
model = joblib.load("../models/random_forest.pkl")
encoders = joblib.load("../models/encoders.pkl")
data_ref = pd.read_csv("../data/cleaned_dataset.csv")

st.set_page_config(page_title="AI Credit Risk System", layout="centered")

st.title("💳 AI Credit Risk & Collections System")
st.write("Predict delinquency and guide business actions")

# Inputs
age = st.slider("Age", 18, 80, 30)
income = st.number_input("Income", 10000, 200000, 50000)
util = st.slider("Credit Utilization", 0.0, 1.0, 0.5)
missed = st.slider("Missed Payments", 0, 10, 1)

if st.button("Analyze Customer"):

    input_data = {
        "Age": age,
        "Income": income,
        "Credit_Utilization": util,
        "Missed_Payments": missed
    }

    df = pd.DataFrame([input_data])

    # Fill missing
    for col in model.feature_names_in_:
        if col not in df.columns:
            if col in data_ref.columns:
                if pd.api.types.is_numeric_dtype(data_ref[col]):
                    df[col] = data_ref[col].median()
                else:
                    df[col] = encoders[col].transform(
                        [data_ref[col].mode()[0]]
                    )[0]
            else:
                df[col] = 0

    # Encode
    for col, le in encoders.items():
        if col in df.columns:
            try:
                df[col] = le.transform(df[col])
            except:
                df[col] = le.transform(
                    [data_ref[col].mode()[0]]
                )[0]

    df = df[model.feature_names_in_]

    # Prediction
    prob = model.predict_proba(df)[0][1]
    prediction = 1 if prob > 0.35 else 0

    # Rule override
    if util > 0.85 and missed >= 3:
        prediction = 1
        prob = max(prob, 0.75)

    # Output
    st.subheader("📊 Risk Assessment")

    if prediction == 1:
        st.error("⚠️ High Risk: Likely Delinquent")
    else:
        st.success("✅ Low Risk: Not Delinquent")

    st.info(f"📈 Risk Probability: {prob:.2f}")

    # Justification
    st.subheader("🧠 Why this prediction?")
    reasons = []

    if util > 0.8:
        reasons.append("High credit utilization")

    if missed >= 3:
        reasons.append("Multiple missed payments")

    if income < 30000:
        reasons.append("Low income")

    if age < 25:
        reasons.append("Younger risk profile")

    if not reasons:
        reasons.append("Stable financial indicators")

    for r in reasons:
        st.write(f"- {r}")

    # Business Actions
    st.subheader("💼 Recommended Actions")

    if prediction == 1:
        st.write("**Immediate:**")
        st.write("- Contact customer within 24 hrs")
        st.write("- Offer repayment plan")

        st.write("**Strategy:**")
        st.write("- Monitor closely")
        st.write("- Reduce credit exposure")

    else:
        st.write("**Preventive:**")
        st.write("- Send reminders")
        st.write("- Maintain engagement")

    # Responsible AI
    st.subheader("⚖️ Responsible AI")
    st.write("- Uses financial behavior only")
    st.write("- Supports human decisions")
    st.write("- Requires fairness monitoring")