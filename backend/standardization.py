import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("data/credit_card.csv")

# Handle missing values
df["MINIMUM_PAYMENTS"] = df["MINIMUM_PAYMENTS"].fillna(
    df["MINIMUM_PAYMENTS"].median()
)

df["CREDIT_LIMIT"] = df["CREDIT_LIMIT"].fillna(
    df["CREDIT_LIMIT"].median()
)

# Select features for clustering
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

# Create feature dataset
X = df[features]

# Create StandardScaler
scaler = StandardScaler()

# Standardize the features
X_scaled = scaler.fit_transform(X)

print("===== STANDARDIZATION COMPLETE =====")

print("\nOriginal shape:")
print(X.shape)

print("\nScaled shape:")
print(X_scaled.shape)

print("\nFirst 5 standardized rows:")
print(X_scaled[:5])

print("\nMean of standardized features:")
print(X_scaled.mean(axis=0))

print("\nStandard deviation of standardized features:")
print(X_scaled.std(axis=0))