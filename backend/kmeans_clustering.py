import pandas as pd
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

# Select features
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

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create K-Means model
kmeans = KMeans(
    n_clusters=2,
    random_state=42,
    n_init=10
)

# Fit model and predict clusters
df["Cluster"] = kmeans.fit_predict(X_scaled)

# Display cluster counts
print("===== K-MEANS CLUSTERING COMPLETE =====")

print("\nNumber of clusters:")
print(kmeans.n_clusters)

print("\nCustomers in each cluster:")
print(df["Cluster"].value_counts().sort_index())

# Display first 10 customers
print("\nFirst 10 customers with cluster labels:")
print(df[["CUST_ID", "Cluster"]].head(10))

# Display cluster centers
print("\nCluster Centers:")
print(kmeans.cluster_centers_)

# Save clustered dataset
df.to_csv("../data/customer_segments.csv", index=False)

print("\nClustered dataset saved successfully!")
print("File: data/customer_segments.csv")