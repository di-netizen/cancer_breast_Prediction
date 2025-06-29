from sklearn.datasets import load_breast_cancer
import pandas as pd

# Load dataset
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Save to CSV
df.to_csv("breast_cancer_data.csv", index=False)

print("✅ Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nTarget value counts:")
print(df['target'].value_counts())
print("0 = malignant, 1 = benign")
