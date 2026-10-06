import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

print("Loading dataset...")

data = pd.read_csv(
    "fraud_data.csv",
    usecols=[
        "type",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "isFraud"
    ],
    nrows=500000
)

print("Dataset loaded!")

# Convert numeric columns
numeric_columns = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest"
]

for column in numeric_columns:
    data[column] = pd.to_numeric(data[column], errors="coerce")

data["isFraud"] = pd.to_numeric(
    data["isFraud"],
    errors="coerce"
)

data = data.dropna()

# Convert transaction type to numbers
encoder = LabelEncoder()
data["type_encoded"] = encoder.fit_transform(data["type"])

# Features
X = data[
    [
        "type_encoded",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest"
    ]
]

# Target
y = data["isFraud"].astype(int)

print("Total transactions:", len(data))
print("Fraud transactions:", y.sum())

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training model...")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# Save model + encoder
joblib.dump(
    {
        "model": model,
        "encoder": encoder
    },
    "model.pkl"
)

print("Model trained successfully!")
print("Model saved as model.pkl")