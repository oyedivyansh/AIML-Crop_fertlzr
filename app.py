import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Load models and scalers
crop_model = joblib.load("crop_recmd_model.sav")
crop_scaler = joblib.load("crop_recmnd_scale.sav")
fertilizer_model = joblib.load("fertilizer_model (1).sav")
fertilizer_scaler = joblib.load("scaler (1).sav")

# Load fertilizer dataset
fert_df = pd.read_csv("Fertilizer Prediction.csv")

# Label encode 'Soil Type' and 'Crop Type' for fertilizer prediction
le_soil = LabelEncoder()
le_crop = LabelEncoder()

fert_df['Soil Type'] = le_soil.fit_transform(fert_df['Soil Type'])
fert_df['Crop Type'] = le_crop.fit_transform(fert_df['Crop Type'])

# Crop labels for crop recommendation output
crop_labels = [
    "rice", "maize", "chickpea", "kidneybeans", "pigeonpeas", "mothbeans",
    "mungbean", "blackgram", "lentil", "pomegranate", "banana", "mango",
    "grapes", "watermelon", "muskmelon", "apple", "orange", "papaya",
    "coconut", "cotton", "jute", "coffee"
]

# Streamlit UI configuration
st.set_page_config(page_title="Crop & Fertilizer Recommender", layout="centered")
st.title("🌱 Crop & Fertilizer Recommendation System")

tab1, tab2 = st.tabs(["🌾 Crop Recommendation", "💊 Fertilizer Recommendation"])

# ================================
# 🌾 CROP RECOMMENDATION TAB
# ================================
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

        try:
            crop_name = crop_labels[int(prediction[0]) - 1]
            st.success(f"✅ Recommended Crop: **{crop_name.capitalize()}**")
        except Exception as e:
            st.error(f"🚨 Could not interpret prediction: {e}")

# ================================
# 💊 FERTILIZER RECOMMENDATION TAB
# ================================
# --- Fertilizer Recommendation Tab ---
with tab2:
    st.header("Fertilizer Recommendation")

    # Show crop options
    crop_options = sorted(fert_df["Crop"].dropna().unique())
    selected_crop = st.selectbox("Select Crop", crop_options)

    N = st.slider("Nitrogen Level (N)", 0, 140, 50, key="fn")
    P = st.slider("Phosphorus Level (P)", 5, 145, 50, key="fp")
    K = st.slider("Potassium Level (K)", 5, 205, 50, key="fk")

    if st.button("🔍 Recommend Fertilizer"):
        try:
            # Step 1: Prepare input DataFrame with correct column names
            input_df = pd.DataFrame([[selected_crop, N, P, K]],
                columns=["Crop", "Nitrogen", "Phosphorous", "Potassium"]
            )

            # Step 2: One-hot encode the 'Crop' column
            input_encoded = pd.get_dummies(input_df, columns=["Crop"])

            # Step 3: Align columns with what the model was trained on
            crop_dummies = pd.get_dummies(fert_df["Crop"])
            expected_columns = crop_dummies.columns.tolist() + ["Nitrogen", "Phosphorous", "Potassium"]
            input_encoded = input_encoded.reindex(columns=expected_columns, fill_value=0)

            # Step 4: Scale and predict
            scaled_input = fertilizer_scaler.transform(input_encoded)
            prediction = fertilizer_model.predict(scaled_input)

            st.success(f"✅ Recommended Fertilizer: **{prediction[0]}**")
        except Exception as e:
            st.error(f"🚨 Error during fertilizer prediction: {e}")




