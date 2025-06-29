import pandas as pd
import joblib

# 1. Load the saved model
model = joblib.load('breast_cancer_rf_model.pkl')

# 2. Simulate a new patient record
#    (values must be scaled the same way as training data)
new_patient = pd.DataFrame([{
    'mean radius': 14.0,
    'mean texture': 20.0,
    'mean perimeter': 90.0,
    'mean area': 600.0,
    'mean smoothness': 0.1,
    'mean compactness': 0.15,
    'mean concavity': 0.1,
    'mean concave points': 0.05,
    'mean symmetry': 0.18,
    'mean fractal dimension': 0.06,
    'radius error': 0.5,
    'texture error': 1.0,
    'perimeter error': 3.0,
    'area error': 40.0,
    'smoothness error': 0.005,
    'compactness error': 0.02,
    'concavity error': 0.02,
    'concave points error': 0.01,
    'symmetry error': 0.02,
    'fractal dimension error': 0.003,
    'worst radius': 16.0,
    'worst texture': 25.0,
    'worst perimeter': 100.0,
    'worst area': 700.0,
    'worst smoothness': 0.12,
    'worst compactness': 0.2,
    'worst concavity': 0.15,
    'worst concave points': 0.08,
    'worst symmetry': 0.2,
    'worst fractal dimension': 0.07
}])

# 3. Predict
prediction = model.predict(new_patient)[0]
probability = model.predict_proba(new_patient)[0][prediction]

# 4. Display result
result = 'Benign (Non-Cancerous)' if prediction == 1 else 'Malignant (Cancerous)'
print(f"🔬 Prediction: {result}")
print(f"✅ Confidence: {probability*100:.2f}%")
