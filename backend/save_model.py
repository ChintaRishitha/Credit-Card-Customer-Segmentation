import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load dataset
df = pd.read_csv("../data/credit_card.csv")

# Handle missing values
df["MINIMUM_PAYMENTS"] = df["MINIMUM_PAYMENTS"].fillna(
    df["MINIMUM_PAYMENTS"].median()
)

df["CREDIT_LIMIT"] = df["CREDIT_LIMIT"].fillna(
    df["CREDIT_LIMIT"].median()
)

# Features used for clustering
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

# Select features
X = df[features]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train final K-Means model
kmeans = KMeans(
    n_clusters=2,
    random_state=42,
    n_init=10
)

kmeans.fit(X_scaled)

# Save scaler
joblib.dump(
    scaler,
    "../models/scaler.pkl"
)

# Save K-Means model
joblib.dump(
    kmeans,
    "../models/kmeans_model.pkl"
)

# Save feature names
joblib.dump(
    features,
    "../models/features.pkl"
)

print("===== MODEL SAVING COMPLETE =====")
print("\nK-Means model saved:")
print("models/kmeans_model.pkl")

print("\nScaler saved:")
print("models/scaler.pkl")

print("\nFeature list saved:")
print("models/features.pkl")

print("\nNumber of clusters:", kmeans.n_clusters)