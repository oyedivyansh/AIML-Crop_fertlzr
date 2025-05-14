import streamlit as st
import pandas as pd
import joblib

# Load models and scalers
crop_model = joblib.load("crop_recmd_model.sav")
crop_scaler = joblib.load("crop_recmnd_scale.sav")
fertilizer_model = joblib.load("fertilizer_model.sav")
fertilizer_scaler = joblib.load("scaler.sav")

# Load fertilizer dataset
fert_df = pd.read_csv("Fertilizer Prediction.csv")

# Find the crop column dynamically
crop_column = None
for col in fert_df.columns:
    if "crop" in col.lower():
        crop_column = col
        break

if crop_column is None:
    st.error("❌ Could not find a crop column in Fertilizer Prediction.csv")
    st.stop()

# Crop label mapping (used for crop model's output)
crop_labels = [
    "rice", "maize", "chickpea", "kidneybeans", "pigeonpeas", "mothbeans",
    "mungbean", "blackgram", "lentil", "pomegranate", "banana", "mango",
    "grapes", "watermelon", "muskmelon", "apple", "orange", "papaya",
    "coconut", "cotton", "jute", "coffee"
]

# Streamlit UI
st.set_page_config(page_title="Crop & Fertilizer Recommender", layout="centered")
st.title("🌱 Crop & Fertilizer Recommendation System")

tab1, tab2 = st.tabs(["🌾 Crop Recommendation", "💊 Fertilizer Recommendation"])

# --- Crop Recommendation Tab ---
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
        scaled_input = crop_scaler.transform(input_data)
        prediction = crop_model.predict(scaled_input)

        # Safely handle the index
        try:
            crop_name = crop_labels[int(prediction[0]) - 1]
            st.success(f"✅ Recommended Crop: **{crop_name.capitalize()}**")
        except Exception as e:
            st.error(f"🚨 Could not interpret prediction: {e}")

# --- Fertilizer Recommendation Tab ---
with tab2:
    st.header("Fertilizer Recommendation")

    crop_options = sorted(fert_df[crop_column].dropna().unique())
    selected_crop = st.selectbox("Select Crop", crop_options)
    fn = st.slider("Nitrogen Level (N)", 0, 140, 50, key="fn")
    fp = st.slider("Phosphorus Level (P)", 5, 145, 50, key="fp")
    fk = st.slider("Potassium Level (K)", 5, 205, 50, key="fk")

    if st.button("🔍 Recommend Fertilizer"):
        try:
            input_df = pd.DataFrame([[selected_crop, fn, fp, fk]], columns=["Crop", "N", "P", "K"])
            input_df["Crop"] = input_df["Crop"].astype(str)

            # One-hot encoding and alignment with model features
            input_encoded = pd.get_dummies(input_df, columns=["Crop"])
            
            # Try to match fertilizer training features (infer from all dummy crops used in the dataset)
            all_crops = pd.get_dummies(fert_df["Crop"].astype(str), prefix="Crop")
            expected_columns = ["N", "P", "K"] + list(all_crops.columns)

            # One-hot encode input and align
            input_encoded = pd.get_dummies(input_df, columns=["Crop"])
            input_encoded = input_encoded.reindex(columns=expected_columns, fill_value=0)


            # Scaling and prediction
            scaled_input = fertilizer_scaler.transform(input_encoded)
            prediction = fertilizer_model.predict(scaled_input)

            st.success(f"✅ Recommended Fertilizer: **{prediction[0]}**")
        except Exception as e:
            st.error(f"🚨 Error during fertilizer prediction: {e}")
