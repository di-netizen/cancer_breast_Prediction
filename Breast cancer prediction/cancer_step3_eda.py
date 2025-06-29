import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load cleaned data
df = pd.read_csv("breast_cancer_data_cleaned.csv")

# 2. Class balance plot
plt.figure(figsize=(6,4))
sns.countplot(x='target', data=df, palette='Set1')
plt.title("Class Distribution: 0 = Malignant, 1 = Benign")
plt.xlabel("Target")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# 3. Correlation heatmap for top 10 features with highest variance
#    (or you can pick features manually)
corr = df.corr().abs()['target'].sort_values(ascending=False)
top_features = corr.index[1:11]  # exclude 'target' itself

plt.figure(figsize=(10,8))
sns.heatmap(df[top_features].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap of Top 10 Features")
plt.tight_layout()
plt.show()

# 4. Distribution plots for a few key features
features_to_plot = ['mean radius', 'mean texture', 'mean perimeter', 'mean area']
for feat in features_to_plot:
    plt.figure(figsize=(6,4))
    sns.kdeplot(data=df, x=feat, hue='target', fill=True, alpha=0.5, palette='Set2')
    plt.title(f"Distribution of {feat} by Class")
    plt.tight_layout()
    plt.show()
