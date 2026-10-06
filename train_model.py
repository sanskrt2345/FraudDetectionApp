import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

print("Loading dataset...")

# Load only required columns and first 200,000 rows
data = pd.read_csv(
    "fraud_data.csv",
    usecols=[
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "isFraud"
    ],
    nrows=200000,
    dtype=str
)

print("Dataset loaded successfully!")

# Convert numeric columns
numeric_columns = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud"
]

for column in numeric_columns:
    data[column] = pd.to_numeric(data[column], errors="coerce")

# Remove invalid rows
data = data.dropna()

print("Data prepared successfully!")
print("Total rows:", len(data))

# Features
X = data[
    [
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest"
    ]
]

# Target
y = data["isFraud"].astype(int)

print("Fraud transactions:", y.sum())
print("Normal transactions:", (y == 0).sum())

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training model...")

# Random Forest
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# Save model
joblib.dump(model, "model.pkl")

print("Model trained successfully!")
print("Model saved as model.pkl")