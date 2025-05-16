import streamlit as st
import pandas as pd
import joblib

# Load models and scalers
crop_model = joblib.load("crop_recmd_model.sav")
crop_scaler = joblib.load("crop_recmnd_scale.sav")
fertilizer_model = joblib.load("fertilizer_model (1).sav")
fertilizer_scaler = joblib.load("scaler (1).sav")

# Load fertilizer dataset
fert_df = pd.read_csv("Fertilizer Prediction.csv")

# Debug: show columns to make sure column names are correct
st.write("Fertilizer dataset columns:", fert_df.columns.tolist())

# Detect correct Crop column name for fertilizer data
crop_col_candidates = ['Crop', 'Crop Type', 'crop', 'crop_type']
fert_crop_col = None
for col in crop_col_candidates:
    if col in fert_df.columns:
        fert_crop_col = col
        break

if fert_crop_col is None:
    st.error("❌ Could not find a 'Crop' column in Fertilizer Prediction.csv. Columns found: " + ", ".join(fert_df.columns))
    st.stop()

# Crop labels for crop recommendation output (index-based)
crop_labels = [
    "rice", "maize", "chickpea", "kidneybeans", "pigeonpeas", "mothbeans",
    "mungbean", "blackgram", "lentil", "pomegranate", "banana", "mango",
    "grapes", "watermelon", "muskmelon", "apple", "orange", "papaya",
    "coconut", "cotton", "jute", "coffee"
]

# Streamlit UI setup
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
        try:
            input_data = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]],
                                      columns=["N", "P", "K", "temperature", "humidity", "ph", "rainfall"])
            scaled_input = crop_scaler.transform(input_data)
            prediction = crop_model.predict(scaled_input)
            crop_index = int(prediction[0]) - 1
            if 0 <= crop_index < len(crop_labels):
                crop_name = crop_labels[crop_index]
                st.success(f"✅ Recommended Crop: **{crop_name.capitalize()}**")
            else:
                st.error("🚨 Prediction index out of range.")
        except Exception as e:
            st.error(f"🚨 Could not make crop recommendation: {e}")

# --- Fertilizer Recommendation Tab ---
with tab2:
    st.header("Fertilizer Recommendation")

    crop_options = sorted(fert_df[fert_crop_col].dropna().unique())
    selected_crop = st.selectbox("Select Crop", crop_options)
    N = st.slider("Nitrogen Level (N)", 0, 140, 50, key="fn")
    P = st.slider("Phosphorus Level (P)", 5, 145, 50, key="fp")
    K = st.slider("Potassium Level (K)", 5, 205, 50, key="fk")

    if st.button("🔍 Recommend Fertilizer"):
        try:
            # Prepare input dataframe with selected crop and fertilizer levels
            input_df = pd.DataFrame([[selected_crop, N, P, K]], columns=[fert_crop_col, "N", "P", "K"])
            input_df[fert_crop_col] = input_df[fert_crop_col].astype(str)

            # One-hot encode the Crop column for input
            input_encoded = pd.get_dummies(input_df, columns=[fert_crop_col])

            # One-hot encode the training fertilizer dataframe's crop column to get expected columns
            fert_crop_dummies = pd.get_dummies(fert_df[fert_crop_col])

            # Expected feature columns = crop dummies + N, P, K
            expected_cols = fert_crop_dummies.columns.tolist() + ["N", "P", "K"]

            # Align input columns to expected columns (add missing cols with 0)
            input_encoded = input_encoded.reindex(columns=expected_cols, fill_value=0)

            # Scale the input features
            scaled_input = fertilizer_scaler.transform(input_encoded)

            # Predict fertilizer recommendation
            prediction = fertilizer_model.predict(scaled_input)

            st.success(f"✅ Recommended Fertilizer: **{prediction[0]}**")

        except Exception as e:
            st.error(f"🚨 Error during fertilizer prediction: {e}")
