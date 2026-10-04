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

# Select numerical columns
numeric_df = df.select_dtypes(include="number")

# Statistical summary
print("===== STATISTICAL SUMMARY =====")
print(numeric_df.describe().T)

# Mean
print("\n===== MEAN =====")
print(numeric_df.mean())

# Median
print("\n===== MEDIAN =====")
print(numeric_df.median())

# Standard deviation
print("\n===== STANDARD DEVIATION =====")
print(numeric_df.std())