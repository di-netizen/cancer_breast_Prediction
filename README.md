# 🩺 Breast Cancer Prediction with Machine Learning

This project uses the **Breast Cancer Wisconsin Diagnostic dataset** to build a machine learning model that classifies tumors as **malignant** or **benign**. The pipeline covers data preparation, exploratory analysis, model training, evaluation, and deployment using a Random Forest classifier.

---

## 📁 Project Structure


---

## 📊 Dataset Overview

- **Source**: `sklearn.datasets.load_breast_cancer()`
- **Samples**: 569
- **Features**: 30 numeric attributes
- **Target**: 0 = Malignant, 1 = Benign

---

## 🧪 Models Trained

- Logistic Regression
- Decision Tree Classifier
- ✅ Random Forest (final model, saved)

Evaluation Metrics:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC Curve & AUC

---

## 📉 Exploratory Data Analysis

- Class distribution visualization
- Correlation heatmap of top features
- Feature distribution (e.g. mean radius, mean area)

---

## 🚀 How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run scripts step by step
python scripts/data_load_script.py
python scripts/data_cleaning.py
python scripts/eda.py
python scripts/model_training.py
python scripts/inference_demo.py
