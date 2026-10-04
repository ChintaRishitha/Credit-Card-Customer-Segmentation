import pandas as pd

# Load dataset
df = pd.read_csv("data/credit_card.csv")

# Handle missing values
df["MINIMUM_PAYMENTS"] = df["MINIMUM_PAYMENTS"].fillna(
    df["MINIMUM_PAYMENTS"].median()
)

df["CREDIT_LIMIT"] = df["CREDIT_LIMIT"].fillna(
    df["CREDIT_LIMIT"].median()
)

# Features selected for customer segmentation
features = [
    "BALANCE",
    "PURCHASES",
    "CASH_ADVANCE",
    "CREDIT_LIMIT",
    "PAYMENTS",
    "PURCHASES_FREQUENCY",
    "CASH_ADVANCE_FREQUENCY",
    "PURCHASES_TRX"
]

X = df[features]

print("===== SELECTED FEATURES =====")
print(features)

print("\nFeature Dataset Shape:")
print(X.shape)

print("\nFirst 5 rows:")
print(X.head())

print("\nMissing Values:")
print(X.isnull().sum())