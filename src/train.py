import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load data
df = pd.read_csv("data/cleaned_dataset.csv")

# Encode categorical columns
encoders = {}
for col in df.select_dtypes(include=["object", "string"]).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Features & target
X = df.drop(columns=["Delinquent_Account", "Customer_ID"])
y = df["Delinquent_Account"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model (balanced for better risk detection)
model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

# Save
joblib.dump(model, "models/random_forest.pkl")
joblib.dump(encoders, "models/encoders.pkl")

print("✅ Model trained and saved")