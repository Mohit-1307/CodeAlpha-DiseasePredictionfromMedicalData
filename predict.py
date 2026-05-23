import numpy as np
import joblib

# Load saved model
model = joblib.load('models/heart_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# ==========================================
# SAMPLE PATIENT DATA
# ==========================================

sample_data = np.array([[
    52,
    1,
    2,
    130,
    250,
    0,
    1,
    187,
    0,
    3.5,
    2,
    0,
    2
]])

# Scale data
sample_data = scaler.transform(sample_data)

# Prediction
prediction = model.predict(sample_data)

# Probability
probability = model.predict_proba(sample_data)

print("\nPrediction Result")

if prediction[0] == 1:
    print("Heart Disease Detected")
else:
    print("No Heart Disease")

print("\nPrediction Probability")
print(probability)