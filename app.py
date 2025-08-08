import streamlit as st
import numpy as np
import joblib


model = joblib.load("diabetes_best_rf_model.pkl")
scaler = joblib.load("scaler.pkl")


st.title("🩺 Diabetes Prediction App")
st.write("Enter your health information to predict the likelihood of diabetes.")
    

pregnancies = st.number_input("Number of Pregnancies", min_value=0)
glucose = st.number_input("Glucose Level", min_value=0)
blood_pressure = st.number_input("Blood Pressure", min_value=0)
skin_thickness = st.number_input("Skin Thickness", min_value=0)
insulin = st.number_input("Insulin Level", min_value=0)
bmi = st.number_input("BMI", min_value=0.0, format="%.1f")
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, format="%.3f")
age = st.number_input("Age", min_value=0)


if st.button("Predict"):
   
    user_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                           insulin, bmi, dpf, age]])
    user_data_scaled = scaler.transform(user_data)

    
    prediction = model.predict(user_data_scaled)[0]
    result = "🔴 Likely Diabetic" if prediction == 1 else "🟢 Not Diabetic"

    st.subheader("Prediction Result:")
    st.success(result)
