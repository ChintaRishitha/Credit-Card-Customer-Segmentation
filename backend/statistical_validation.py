import pandas as pd
from scipy.stats import mannwhitneyu

# Load clustered dataset
df = pd.read_csv("../data/customer_segments.csv")

# Features to test
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

# Separate the two clusters
cluster_0 = df[df["Cluster"] == 0]
cluster_1 = df[df["Cluster"] == 1]

print("===== STATISTICAL VALIDATION =====")

print("\nComparing Cluster 0 and Cluster 1")
print("Test: Mann-Whitney U")

results = []

for feature in features:

    # Perform Mann-Whitney U test
    statistic, p_value = mannwhitneyu(
        cluster_0[feature],
        cluster_1[feature],
        alternative="two-sided"
    )

    significant = "Yes" if p_value < 0.05 else "No"

    results.append({
        "Feature": feature,
        "U Statistic": statistic,
        "P-Value": p_value,
        "Significant": significant
    })

# Create results dataframe
results_df = pd.DataFrame(results)

# Display results
print("\nStatistical Test Results:")
print(results_df.to_string(index=False))

# Save results
results_df.to_csv(
    "../data/statistical_validation.csv",
    index=False
)

print("\nSignificance level: 0.05")

print("\nStatistical validation results saved successfully!")
print("File: data/statistical_validation.csv")