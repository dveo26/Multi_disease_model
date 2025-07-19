# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 20:06:53 2025

@author: devon
"""
import pickle
import streamlit as st
from streamlit_option_menu import option_menu

# --- MODEL AND SCALER LOADING ---
# Make sure the paths to your .sav files are correct
try:
    diabetes_model = pickle.load(open('diabities_model.sav', 'rb'))
    heart_disease_model = pickle.load(open('heart_disease_model.sav', 'rb'))
    diabetes_scaler = pickle.load(open('diabities_scaler.sav', 'rb'))
    heart_scaler = pickle.load(open('heart_scaler.sav', 'rb'))
except FileNotFoundError:
    st.error("Fatal Error: A model or scaler file was not found. Please ensure all .sav files are in the correct directory.")
    st.stop()


# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    selected = option_menu('Disease Prediction System',
                           ['Diabetes Prediction', 'Heart Disease Prediction'],
                           icons=['activity', 'heart-pulse'],
                           default_index=0)


# --- DIABETES PREDICTION PAGE ---
if selected == 'Diabetes Prediction':

    st.title('Diabetes Prediction')
    st.write("Enter the patient's details below to predict the likelihood of diabetes.")

    # Create columns for input fields
    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.number_input('Number of Pregnancies', min_value=0, step=1)
    with col2:
        Glucose = st.number_input('Glucose Level (mg/dL)', min_value=0)
    with col3:
        BloodPressure = st.number_input('Blood Pressure (mm Hg)', min_value=0)
    with col1:
        SkinThickness = st.number_input('Skin Thickness (mm)', min_value=0)
    with col2:
        Insulin = st.number_input('Insulin Level (mu U/ml)', min_value=0)
    with col3:
        BMI = st.number_input('BMI value')
    with col1:
        DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function')
    with col2:
        Age = st.number_input('Age (years)', min_value=0, step=1)

    # Prediction button
    if st.button('Predict Diabetes'):
        input_data = [[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]]
        
        # ✅ Scale the input data using the diabetes scaler
        scaled_data = diabetes_scaler.transform(input_data)
        
        # Predict on the scaled data
        prediction = diabetes_model.predict(scaled_data)

        st.subheader('Prediction Result')
        if prediction[0] == 1:
            st.error('The model predicts that the person **is diabetic**.')
        else:
            st.success('The model predicts that the person **is not diabetic**.')


# --- HEART DISEASE PREDICTION PAGE ---
if selected == 'Heart Disease Prediction':

    st.title('Heart Disease Prediction')
    st.write("Enter the patient's details below to predict the likelihood of heart disease.")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input('Age', min_value=0, max_value=120, step=1)
    with col2:
        sex_option = st.selectbox('Sex', ('Male', 'Female'))
        sex = 1 if sex_option == 'Male' else 0
    with col3:
        cp_options = {0: 'Typical Angina', 1: 'Atypical Angina', 2: 'Non-anginal Pain', 3: 'Asymptomatic'}
        cp_selected = st.selectbox('Chest Pain Type', options=list(cp_options.values()))
        chest_pain_type = list(cp_options.keys())[list(cp_options.values()).index(cp_selected)]
    with col1:
        resting_bp_s = st.number_input('Resting Blood Pressure (mm Hg)', min_value=0)
    with col2:
        cholesterol = st.number_input('Cholesterol (mg/dl)', min_value=0)
    with col3:
        fbs_option = st.selectbox('Fasting Blood Sugar > 120 mg/dl', ('No', 'Yes'))
        fasting_blood_sugar = 1 if fbs_option == 'Yes' else 0
    with col1:
        ecg_options = {0: 'Normal', 1: 'ST-T wave abnormality', 2: 'Left ventricular hypertrophy'}
        ecg_selected = st.selectbox('Resting ECG', options=list(ecg_options.values()))
        resting_ecg = list(ecg_options.keys())[list(ecg_options.values()).index(ecg_selected)]
    with col2:
        max_heart_rate = st.number_input('Maximum Heart Rate Achieved', min_value=0)
    with col3:
        exang_option = st.selectbox('Exercise Induced Angina', ('No', 'Yes'))
        exercise_angina = 1 if exang_option == 'Yes' else 0
    with col1:
        oldpeak = st.number_input('Oldpeak (ST depression)')
    with col2:
        slope_options = {0: 'Upsloping', 1: 'Flat', 2: 'Downsloping'}
        slope_selected = st.selectbox('ST Slope', options=list(slope_options.values()))
        st_slope = list(slope_options.keys())[list(slope_options.values()).index(slope_selected)]

    # Prediction button
    if st.button('Predict Heart Disease'):
        input_data = [[age, sex, chest_pain_type, resting_bp_s, cholesterol, fasting_blood_sugar, resting_ecg, max_heart_rate, exercise_angina, oldpeak, st_slope]]
        
        # ✅ Scale the input data using the heart disease scaler
        scaled_data = heart_scaler.transform(input_data)
        
        # Predict on the scaled data
        prediction = heart_disease_model.predict(scaled_data)

        st.subheader('Prediction Result')
        if prediction[0] == 1:
            st.error('The model predicts that the person **has heart disease**.')
        else:
            st.success('The model predicts that the person **does not have heart disease**.')