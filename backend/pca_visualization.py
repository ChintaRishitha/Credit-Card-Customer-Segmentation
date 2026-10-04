import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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

# Select features
X = df[features]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Create PCA dataframe
pca_df = pd.DataFrame(
    X_pca,
    columns=["PC1", "PC2"]
)

# Add cluster labels
pca_df["Cluster"] = df["Cluster"]

# Print explained variance
print("===== PCA ANALYSIS =====")

print("\nExplained variance ratio:")
print(pca.explained_variance_ratio_)

print("\nTotal variance explained:")
print(
    round(pca.explained_variance_ratio_.sum() * 100, 2),
    "%"
)

# Plot clusters
plt.figure(figsize=(10, 7))

for cluster in sorted(pca_df["Cluster"].unique()):

    cluster_data = pca_df[
        pca_df["Cluster"] == cluster
    ]

    plt.scatter(
        cluster_data["PC1"],
        cluster_data["PC2"],
        label=f"Cluster {cluster}",
        alpha=0.6
    )

plt.title("Customer Segmentation using PCA")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.grid(True)

plt.tight_layout()

# Save graph
plt.savefig("../pca_clusters.png", dpi=300)

# Display graph
plt.show()