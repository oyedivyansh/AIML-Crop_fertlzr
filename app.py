import streamlit as st
import pandas as pd
import joblib

# Load models and scalers
crop_model = joblib.load("crop_recmd_model.sav")
crop_scaler = joblib.load("scaler_crop.sav")

fertilizer_model = joblib.load("fertilizer_model.sav")
fertilizer_scaler = joblib.load("scaler.sav")

# Load fertilizer dataset to get crop names
fert_df = pd.read_csv("Fertilizer Prediction.csv")
crop_options = sorted(fert_df["Crop"].unique())

st.set_page_config(page_title="Smart Agri Assistant", layout="centered")
st.title("🌾 Crop & Fertilizer Recommendation System")

tab1, tab2 = st.tabs(["🌿 Crop Recommendation", "🧪 Fertilizer Recommendation"])

# 🌿 CROP RECOMMENDATION TAB
with tab1:
    st.subheader("🌱 Input Soil & Weather Data for Crop Recommendation")

    N = st.slider("Nitrogen (N)", 0, 140, 50)
    P = st.slider("Phosphorus (P)", 5, 145, 50)
    K = st.slider("Potassium (K)", 5, 205, 50)
    temperature = st.number_input("Temperature (°C)", 0.0, 60.0, 25.0)
    humidity = st.number_input("Humidity (%)", 0.0, 100.0, 50.0)
    ph = st.number_input("pH", 0.0, 14.0, 6.5)
    rainfall = st.number_input("Rainfall (mm)", 0.0, 500.0, 100.0)

    if st.button("🌿 Recommend Crop"):
        input_data = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]],
                                  columns=["N", "P", "K", "temperature", "humidity", "ph", "rainfall"])
        input_scaled = crop_scaler.transform(input_data)
        prediction = crop_model.predict(input_scaled)
        st.success(f"✅ Recommended Crop: **{prediction[0].capitalize()}**")

# 🧪 FERTILIZER RECOMMENDATION TAB
with tab2:
    st.subheader("🌾 Input Soil Nutrients for Fertilizer Recommendation")

    selected_crop = st.selectbox("Select Crop", crop_options)
    N = st.slider("Nitrogen Level (N)", 0, 140, 50, key="fn")
    P = st.slider("Phosphorus Level (P)", 5, 145, 50, key="fp")
    K = st.slider("Potassium Level (K)", 5, 205, 50, key="fk")

    if st.button("🧪 Recommend Fertilizer"):
        # Match crop name to numeric encoding if your model expects it
        crop_row = fert_df[fert_df["Crop"] == selected_crop].iloc[0]
        crop_code = crop_row["Crop_code"] if "Crop_code" in crop_row else crop_options.index(selected_crop)

        input_data = pd.DataFrame([[crop_code, N, P, K]], columns=["Crop", "N", "P", "K"])
        input_scaled = fertilizer_scaler.transform(input_data)
        prediction = fertilizer_model.predict(input_scaled)
        st.success(f"✅ Recommended Fertilizer: **{prediction[0]}**")
