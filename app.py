import streamlit as st
import pandas as pd
import joblib

# ====== SIDEBAR: Branding and About Section ======
with st.sidebar:
    # Logo line removed as requested
    st.markdown("""
        ### About This App
        *Developed by **Sahil Bagde**  
        Data Scientist | ML Enthusiast  
        Email: bagdesahil31@example.com*
        [LinkedIn](https://www.linkedin.com/in/sahil-bagde-9373a724a/)

        ---
        #### Disclaimer
        This app is for general informational purposes only, not medical advice.
        Please consult a licensed medical professional for any health decisions.
    """)

# ====== MAIN TITLE & DESCRIPTION ======
st.title("🫀 Heart Stroke Risk Predictor")
st.markdown("""
    Provide your health details to estimate your risk of heart disease.
    **All inputs are strictly confidential and for demonstration only.**
""")

# ====== Load Model/Scaler/Columns ======
@st.cache_resource
def load_assets():
    model = joblib.load("knn_heart_model.pkl")
    scaler = joblib.load("heart_scaler.pkl")
    expected_columns = joblib.load("heart_column.pkl")
    return model, scaler, expected_columns
try:
    model, scaler, expected_columns = load_assets()
except Exception as e:
    st.error("Model files not found or corrupted. Please ensure .pkl files are present.")
    st.stop()

# ====== USER INPUT FORM ======
with st.form("heart_form"):
    age = st.slider("Age", 18, 100, 40, help="Your age in years.")
    sex = st.selectbox("Sex", ["M", "F"], help="M: Male | F: Female")
    chest_pain = st.selectbox(
        "Chest Pain Type", 
        ["ATA", "NAP", "TA", "ASY"], 
        help="ATA: Atypical Angina, NAP: Non-anginal Pain, TA: Typical Angina, ASY: Asymptomatic"
    )
    resting_bp = st.number_input(
        "Resting Blood Pressure (mm Hg)", 80, 200, 120,
        help="Usual resting systolic blood pressure (mm Hg)"
    )
    cholesterol = st.number_input(
        "Cholesterol (mg/dL)", 100, 600, 200, help="Serum cholesterol in mg/dL"
    )
    fasting_bs = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dL", [0, 1], help="1: Yes, 0: No"
    )
    resting_ecg = st.selectbox(
        "Resting ECG", ["Normal", "ST", "LVH"], 
        help="Normal, ST: having ST-T wave abnormality, LVH: left ventricular hypertrophy"
    )
    max_hr = st.slider(
        "Max Heart Rate", 60, 220, 150, help="Max heart rate achieved during test"
    )
    exercise_angina = st.selectbox(
        "Exercise-Induced Angina", ["Y", "N"],
        help="Y: Yes, N: No. Chest pain during exercise."
    )
    oldpeak = st.slider(
        "Oldpeak (ST Depression)", 0.0, 6.0, 1.0, help="ST depression induced by exercise."
    )
    st_slope = st.selectbox(
        "ST Slope", ["Up", "Flat", "Down"], help="Slope of the peak exercise ST segment"
    )

    submitted = st.form_submit_button("Predict Heart Stroke Risk")

# ====== PROCESS & PREDICT ======
if submitted:
    features = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        "Sex_" + sex: 1,
        "ChestPainType_" + chest_pain: 1,
        "RestingECG_" + resting_ecg: 1,
        "ExerciseAngina_" + exercise_angina: 1,
        "ST_Slope_" + st_slope: 1
    }
    input_df = pd.DataFrame([features])
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[expected_columns]
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]
    st.markdown("### Prediction Result")
    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease")
        st.info("Consult with a licensed cardiologist for professional guidance.")
    else:
        st.success("✅ Low Risk of Heart Disease")
        st.info("Maintain a healthy lifestyle for continued wellness!")
    with st.expander("Show input summary"):
        st.write(input_df)

# ====== Professional Footer ======
st.markdown("""
---
*Developed with ❤️ by Sahil Bagde — 2025*
""")
