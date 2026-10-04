import pandas as pd

# Load dataset
df = pd.read_csv("data/credit_card.csv")

print("Original Shape:", df.shape)

# Fill missing numerical values with median
df["MINIMUM_PAYMENTS"] = df["MINIMUM_PAYMENTS"].fillna(
    df["MINIMUM_PAYMENTS"].median()
)

df["CREDIT_LIMIT"] = df["CREDIT_LIMIT"].fillna(
    df["CREDIT_LIMIT"].median()
)

print("\nMissing values after cleaning:")
print(df.isnull().sum())

print("\nFinal Shape:", df.shape)