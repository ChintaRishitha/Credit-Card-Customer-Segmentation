import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("data/credit_card.csv")

# Handle missing values
df["MINIMUM_PAYMENTS"] = df["MINIMUM_PAYMENTS"].fillna(
    df["MINIMUM_PAYMENTS"].median()
)

df["CREDIT_LIMIT"] = df["CREDIT_LIMIT"].fillna(
    df["CREDIT_LIMIT"].median()
)

# Remove customer ID
numeric_df = df.drop(columns=["CUST_ID"])

# Create correlation matrix
correlation = numeric_df.corr()

# Display heatmap
plt.figure(figsize=(14, 10))

sns.heatmap(
    correlation,
    annot=False,
    cmap="coolwarm",
    linewidths=0.5
)

plt.title("Credit Card Customer Feature Correlation")
plt.tight_layout()

# Save image
plt.savefig("correlation_heatmap.png", dpi=300)

# Show graph
plt.show()