import pandas as pd
from sklearn.preprocessing import StandardScaler

# 1. Load the CSV
df = pd.read_csv("breast_cancer_data.csv")

# 2. Check for missing values
print("🔍 Missing values per column:\n", df.isnull().sum())

# 3. Separate features and target
X = df.drop("target", axis=1)
y = df["target"]

# 4. Standardize all features
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# 5. Combine back with target
df_clean = X_scaled.copy()
df_clean["target"] = y

# 6. Save cleaned & scaled dataset
df_clean.to_csv("breast_cancer_data_cleaned.csv", index=False)
print("\n✅ Cleaned data saved as 'breast_cancer_data_cleaned.csv'")
print(df_clean.head())
