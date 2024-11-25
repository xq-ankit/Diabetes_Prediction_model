import streamlit as st
import numpy as np
import joblib


model = joblib.load('diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')


st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺", layout="wide")
st.title("Diabetes Prediction App")

#sidebar
st.sidebar.header("Patient Information")
st.sidebar.markdown("Please enter the details below to predict diabetes risk.")
pregnancies = st.sidebar.number_input("Pregnancies", min_value=0, value=0)
glucose = st.sidebar.number_input("Glucose Level", min_value=0.0, value=120.0)
blood_pressure = st.sidebar.number_input("Blood Pressure", min_value=0.0, value=70.0)
skin_thickness = st.sidebar.number_input("Skin Thickness", min_value=0.0, value=20.0)
insulin = st.sidebar.number_input("Insulin Level", min_value=0.0, value=80.0)
bmi = st.sidebar.number_input("BMI", min_value=0.0, value=25.0)
diabetes_pedigree_function = st.sidebar.number_input("Diabetes Pedigree Function", min_value=0.0, value=0.0)
age = st.sidebar.number_input("Age", min_value=0, value=30)

st.markdown("""
    <style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        border-radius: 10px;
        padding: 10px 24px;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    </style>
    """, unsafe_allow_html=True)


if st.sidebar.button("Predict"):
    input_data = scaler.transform([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]])
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.markdown("<h3 style='color: red;'>The patient is likely to have diabetes.</h3>", unsafe_allow_html=True)
    else:
        st.markdown("<h3 style='color: green;'>The patient is unlikely to have diabetes.</h3>", unsafe_allow_html=True)


st.markdown("""
    **Important Notes:**
    - Glucose Level refers to the concentration of glucose in the blood.
    - Blood Pressure indicates the force of blood against the walls of your arteries.
    - Skin Thickness is measured at the tricep area to estimate body fat.
    - Insulin level shows the amount of insulin in the blood.
    - BMI is a measure of body fat based on height and weight.
    - Diabetes Pedigree Function indicates the genetic predisposition to diabetes.
    - Age plays a crucial role in the likelihood of diabetes.
""")
