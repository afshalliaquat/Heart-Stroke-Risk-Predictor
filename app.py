import streamlit as st
import pandas as pd
import joblib

# Load model and preprocessing objects
model = joblib.load("KNN_heart.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")

st.markdown("""
    <style>
    .main .block-container {
        max-width: 95% !important;
        padding-left: 3rem;
        padding-right: 3rem;
    }

    .stApp {
        background: linear-gradient(135deg, #e3f2fd, #ffffff); 
        padding: 2rem;
        color: #333333;
    }

    body {
        color: #333333;
    }

    h1, h2, h3, h4 {
        font-size: 28px !important;
        color: #2c3e50;
    }

    label, .stTextInput label, .stSelectbox label, .stSlider label, .stNumberInput label {
        font-size: 18px !important;
        color: #2c3e50 !important;
    }

    .main {
        background-color: rgba(255, 255, 255, 0.8);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    }

    input[type="text"], textarea, .stTextInput input, .stSelectbox div[data-baseweb="select"] {
        background-color: white !important;
        color: #333333 !important;
        border-radius: 8px !important;
        border: 1px solid #cccccc !important;
    }

    .stSlider > div {
        color: #42a5f5;
    }

    .stButton > button {
        background-color: #42a5f5;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.5em 1.5em;
        font-size: 18px;
    }

    .stButton > button:hover {
        background-color: #1e88e5;
        transition: 0.3s;
    }

    .css-1cpxqw2, .css-1offfwp {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 0 8px rgba(0,0,0,0.04);
    }
    </style>
""", unsafe_allow_html=True)


# --- Page Title ---
st.title("💓 Heart Stroke Risk Predictor")
st.markdown("<h4 style='color:#d6336c;'>Check your heart's health status with just a few inputs.</h4>", unsafe_allow_html=True)

st.set_page_config(page_title="Heart Risk Predictor", page_icon="❤️", layout="centered")

# --- User Inputs ---
age = st.slider("🧓 Age", 18, 100, 40)
sex = st.selectbox("⚥ Sex", ["M", "F"])
chest_pain = st.selectbox("💢 Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
resting_bp = st.number_input("💉 Resting Blood Pressure (mm Hg)", 80, 200, 120)
cholesterol = st.number_input("🍔 Cholesterol (mg/dL)", 100, 600, 200)
fasting_bs = st.selectbox("🩸 Fasting Blood Sugar > 120 mg/dL", [0, 1])
resting_ecg = st.selectbox("🧪 Resting ECG", ["Normal", "ST", "LVH"])
max_hr = st.slider("❤️ Max Heart Rate", 60, 220, 150)
exercise_angina = st.selectbox("🏃 Exercise-Induced Angina", ["Y", "N"])
oldpeak = st.slider("📉 Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
st_slope = st.selectbox("📈 ST Slope", ["Up", "Flat", "Down"])

# --- Prediction ---
if st.button("🔍 Predict"):

    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngina_' + exercise_angina: 1,
        'ST_Slope_' + st_slope: 1
    }

    input_df = pd.DataFrame([raw_input])

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]

    st.markdown("---")

    if prediction == 1:
        st.error("🚨 **High Risk of Heart Disease!**\n\nPlease consult a doctor for further diagnosis.")
    else:
        st.success("✅ **Low Risk of Heart Disease**\n\nKeep up the healthy lifestyle!")

# Footer
st.markdown("""
    <hr>
    <p style='text-align:center; color: #888;'>Developed by <b>Afshal</b> | Powered by <i>Streamlit</i></p>
""", unsafe_allow_html=True)
