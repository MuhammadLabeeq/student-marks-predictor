import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Student Marks Predictor", page_icon="🎓")

# Load pre-trained model and medians
@st.cache_resource
def load_assets():
    model = joblib.load("student_model.pkl")
    medians = joblib.load("medians.pkl")
    return model, medians

model, medians = load_assets()

st.title("🎓 Student Final Marks Predictor")
st.write("Predict student final marks using the pre-trained Random Forest model.")

# User Inputs
st.sidebar.header("Input Student Details")
study_hours = st.sidebar.number_input("Study Hours per week", min_value=0.0, max_value=168.0, value=12.0, step=0.5)
attendance = st.sidebar.number_input("Attendance percentage (%)", min_value=0.0, max_value=100.0, value=85.0, step=1.0)
previous_marks = st.sidebar.number_input("Previous marks (0-100)", min_value=0.0, max_value=100.0, value=75.0, step=1.0)
sleep_hours = st.sidebar.number_input("Sleep hours per night", min_value=0.0, max_value=24.0, value=7.0, step=0.5)

if st.sidebar.button("Predict Score"):
    new_student = pd.DataFrame({
        'study_hours_per_week': [study_hours],
        'attendance_pct': [attendance],
        'previous_marks': [previous_marks],
        'sleep_hours_per_night': [sleep_hours]
    })

    # Fill any potential missing values using medians
    new_student = new_student.fillna(medians)

    predicted_score = model.predict(new_student)[0]
    predicted_score = float(np.clip(predicted_score, 0, 100))

    st.subheader("Prediction Result")
    st.metric(label="Predicted Final Marks", value=f"{predicted_score:.1f} / 100")