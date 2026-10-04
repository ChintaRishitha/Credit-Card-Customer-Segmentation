import pandas as pd

# Load clustered dataset
df = pd.read_csv("../data/customer_segments.csv")

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

# Calculate mean values for each cluster
cluster_summary = df.groupby("Cluster")[features].mean()

print("===== CUSTOMER SEGMENT ANALYSIS =====")

print("\nAverage values for each cluster:")
print(cluster_summary.round(2))

# Calculate median values
cluster_median = df.groupby("Cluster")[features].median()

print("\nMedian values for each cluster:")
print(cluster_median.round(2))

# Customer count and percentage
cluster_counts = df["Cluster"].value_counts().sort_index()

cluster_percentage = (
    cluster_counts / len(df) * 100
).round(2)

print("\nCustomer count by cluster:")
print(cluster_counts)

print("\nCustomer percentage by cluster:")
print(cluster_percentage)

# Save summary
cluster_summary.to_csv("../data/cluster_summary.csv")

print("\nCluster summary saved successfully!")
print("File: data/cluster_summary.csv")