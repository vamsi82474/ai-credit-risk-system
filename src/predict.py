import joblib
import pandas as pd

model = joblib.load("models/random_forest.pkl")
encoders = joblib.load("models/encoders.pkl")
data_ref = pd.read_csv("data/cleaned_dataset.csv")

def predict(data_dict):
    df = pd.DataFrame([data_dict])

    # Fill missing columns
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

    prob = model.predict_proba(df)[0][1]

    # Lower threshold
    prediction = 1 if prob > 0.35 else 0

    # Rule override
    if data_dict.get("Credit_Utilization", 0) > 0.85 and data_dict.get("Missed_Payments", 0) >= 3:
        prediction = 1
        prob = max(prob, 0.75)

    return prediction, prob


# Test sample
sample = {
    "Age": 25,
    "Income": 20000,
    "Credit_Utilization": 0.9,
    "Missed_Payments": 5
}

pred, prob = predict(sample)
print("Delinquent" if pred == 1 else "Not Delinquent", "| Prob:", prob)