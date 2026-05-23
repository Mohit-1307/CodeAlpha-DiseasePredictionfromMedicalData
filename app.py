import streamlit as st
import numpy as np
import joblib

# ==========================================
# LOAD MODEL
# ==========================================

model = joblib.load('models/heart_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title='Heart Disease Prediction',
    page_icon='❤️',
    layout='centered'
)

# ==========================================
# TITLE
# ==========================================

st.title('❤️ Heart Disease Prediction System')

st.write(
    'Enter patient medical information to predict heart disease risk.'
)

# ==========================================
# INPUT FIELDS
# ==========================================

age = st.slider('Age', 20, 100, 50)

sex = st.selectbox(
    'Sex',
    [0, 1],
    format_func=lambda x: 'Female' if x == 0 else 'Male'
)

cp = st.slider('Chest Pain Type', 0, 3)

trestbps = st.slider('Resting Blood Pressure', 80, 200, 120)

chol = st.slider('Cholesterol', 100, 600, 250)

fbs = st.selectbox('Fasting Blood Sugar', [0, 1])

restecg = st.selectbox('Rest ECG', [0, 1, 2])

thalach = st.slider('Maximum Heart Rate', 60, 220, 150)

exang = st.selectbox('Exercise Induced Angina', [0, 1])

oldpeak = st.slider('Old Peak', 0.0, 6.0, 1.0)

slope = st.selectbox('Slope', [0, 1, 2])

ca = st.selectbox('Number of Major Vessels', [0, 1, 2, 3, 4])

thal = st.selectbox('Thalassemia', [0, 1, 2, 3])

# ==========================================
# PREDICTION BUTTON
# ==========================================

if st.button('Predict Disease'):

    input_data = np.array([[
        age,
        sex,
        cp,
        trestbps,
        chol,
        fbs,
        restecg,
        thalach,
        exang,
        oldpeak,
        slope,
        ca,
        thal
    ]])

    # Scale data
    input_data = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_data)

    # Probability
    probability = model.predict_proba(input_data)

    st.subheader('Prediction Result')

    if prediction[0] == 1:
        st.error('Heart Disease Detected')
    else:
        st.success('No Heart Disease Detected')

    st.subheader('Prediction Probability')

    st.write(probability)