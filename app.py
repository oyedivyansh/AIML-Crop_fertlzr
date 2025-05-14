import streamlit as st
import joblib
import pandas as pd

# Load models and scalers
crop_model = joblib.load("crop_recmd_model.sav")
crop_scaler = joblib.load("crop_recmnd_scale.sav")
fertilizer_model = joblib.load("fertilizer_model.sav")
fertilizer_scaler = joblib.load("scaler.sav")

# Load fertilizer dataset to get crop options
fert_df = pd.read_csv("Fertilizer Prediction.csv")
crop_options = sorted(fert_df["Crop"].unique())

# Expected one-hot encoded column names used in fertilizer model
fertilizer_model_columns = ['N', 'P', 'K'] + [f'Crop_{crop}' for crop in crop_options]

# Streamlit page config
st.set_page_config(page_title="Crop & Fertilizer Recommender", layout="centered")
st.title("🌱 Crop & Fertilizer Recommendation System")

# Tabs
tab1, tab2 = st.tabs(["🌾 Crop Recommendation", "💊 Fertilizer Recommendation"])

# ----- Crop Recommendation -----
with tab1:
    st.header("Crop Recommendation")

    N = st.slider("Nitrogen (N)", 0, 140, 50)
    P = st.slider("Phosphorus (P)", 5, 145, 50)
    K = st.slider("Potassium (K)", 5, 205, 50)
    temperature = st.number_input("Temperature (°C)", 0.0, 60.0, 25.0)
    humidity = st.number_input("Humidity (%)", 0.0, 100.0, 50.0)
    ph = st.number_input("pH", 0.0, 14.0, 6.5)
    rainfall = st.number_input("Rainfall (mm)", 0.0, 500.0, 100.0)

    if st.button("🔍 Recommend Crop"):
        input_data = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]],
                                  columns=["N", "P", "K", "temperature", "humidity", "ph", "rainfall"])
        input_scaled = crop_scaler.transform(input_data)
        prediction = crop_model.predict(input_scaled)

        st.success(f"✅ Recommended Crop: **{prediction[0].capitalize()}**")

# ----- Fertilizer Recommendation -----
with tab2:
    st.header("Fertilizer Recommendation")

    crop_name = st.selectbox("Select Crop", crop_options)
    fn = st.slider("Nitrogen Level (N)", 0, 140, 50, key="fn")
    fp = st.slider("Phosphorus Level (P)", 5, 145, 50, key="fp")
    fk = st.slider("Potassium Level (K)", 5, 205, 50, key="fk")

    if st.button("🔍 Recommend Fertilizer"):
        # Prepare input
        input_data = pd.DataFrame([[crop_name, fn, fp, fk]], columns=["Crop", "N", "P", "K"])

        # One-hot encode 'Crop'
        input_encoded = pd.get_dummies(input_data, columns=["Crop"])

        # Add missing columns and ensure order
        for col in fertilizer_model_columns:
            if col not in input_encoded:
                input_encoded[col] = 0
        input_encoded = input_encoded[fertilizer_model_columns]

        # Scale and predict
        scaled_input = fertilizer_scaler.transform(input_encoded)
        prediction = fertilizer_model.predict(scaled_input)

        st.success(f"✅ Recommended Fertilizer: **{prediction[0]}**")
