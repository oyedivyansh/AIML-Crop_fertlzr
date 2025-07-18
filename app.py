# app.py
import streamlit as st
import pandas as pd
import pickle
import os

# --- Streamlit UI Configuration (MUST BE FIRST STREAMLIT COMMAND) ---
st.set_page_config(
    page_title="Smart Farm Recommendation System",
    page_icon="🌾", # A nice emoji icon for the browser tab
    layout="centered", # Or "wide" for more horizontal space
    initial_sidebar_state="auto" # Controls sidebar initial state
)

# --- Configuration for file paths ---
# These paths are relative to where app.py is located
CROP_MODEL_PATH = os.path.join('models', 'crop_model.pkl')
FERTILIZER_MODEL_PATH = os.path.join('models', 'fertilizer_model.pkl')
DATA_PATH = os.path.join('data', 'Fertilizer_Prediction.csv') # Updated file name

# --- Cached Loading Functions ---
# @st.cache_resource is used for models and large objects that don't change often.
# It ensures models are loaded only once across multiple user interactions.
@st.cache_resource
def load_model_and_metadata(model_path):
    """
    Loads a trained machine learning model, its label encoders, and feature columns.
    Includes robust error handling for missing files.
    """
    try:
        if not os.path.exists(model_path):
            st.error(f"Error: Model file not found at '{model_path}'.")
            st.error("Please ensure you have run 'train_models.py' script to train and save the models.")
            st.stop() # Stop the app if a crucial model file is missing

        with open(model_path, 'rb') as file:
            loaded_data = pickle.load(file)
        return loaded_data['model'], loaded_data['label_encoders'], loaded_data['feature_columns']
    except Exception as e:
        st.error(f"An error occurred while loading model '{os.path.basename(model_path)}': {e}")
        st.info("Check if 'train_models.py' ran successfully and the .pkl files are valid.")
        st.stop()

# @st.cache_data is used for dataframes or other data that doesn't change often.
# It caches the data loading, preventing re-reading the CSV on every interaction.
@st.cache_data
def load_raw_data_for_unique_features():
    """
    Loads raw data specifically to extract unique crop types and soil types for Streamlit dropdowns.
    Includes robust error handling for missing data file.
    """
    try:
        if not os.path.exists(DATA_PATH):
            st.error(f"Error: Data file not found at '{DATA_PATH}'.")
            st.error("Please ensure 'Fertilizer_Prediction.csv' is placed inside the 'data/' folder.")
            st.stop()
        df_raw = pd.read_csv(DATA_PATH)
        # Correct column names from the raw CSV
        df_raw.rename(columns={
            'Temparature': 'Temperature',
            'Phosphorous': 'Phosphorus'
        }, inplace=True)

        unique_crops = []
        if 'Crop Type' in df_raw.columns:
            unique_crops = df_raw['Crop Type'].unique().tolist()
        else:
            st.error("Error: 'Crop Type' column not found in your 'Fertilizer_Prediction.csv'.")
            st.stop()

        unique_soil_types = []
        if 'Soil Type' in df_raw.columns:
            unique_soil_types = df_raw['Soil Type'].unique().tolist()
        else:
            st.error("Error: 'Soil Type' column not found in your 'Fertilizer_Prediction.csv'.")
            st.stop()

        return unique_crops, unique_soil_types
    except Exception as e:
        st.error(f"An error occurred while loading raw data for dropdowns: {e}")
        st.info("Check if 'Fertilizer_Prediction.csv' is correctly formatted and accessible.")
        st.stop()

# --- Load Models, Encoders, and Unique Feature Values at App Startup ---
# Load Crop Recommendation Model
crop_model, crop_label_encoders, crop_feature_columns = load_model_and_metadata(CROP_MODEL_PATH)
crop_type_le_for_crop_model = crop_label_encoders.get('Crop Type')
soil_type_le_for_crop_model = crop_label_encoders.get('Soil Type')

# Load Fertilizer Recommendation Model
fert_model, fert_label_encoders, fert_feature_columns = load_model_and_metadata(FERTILIZER_MODEL_PATH)
fertilizer_le = fert_label_encoders.get('Fertilizer Name')
crop_type_le_for_fert_model = fert_label_encoders.get('Crop Type')
soil_type_le_for_fert_model = fert_label_encoders.get('Soil Type')

# Get unique crop types and soil types for dropdowns
unique_crop_types, unique_soil_types = load_raw_data_for_unique_features()

st.title("🌾 Smart Farm Recommendation System")
st.markdown("""
    This application offers two AI-powered recommendations:
    1.  **Crop Recommendation:** Suggests the best crop for your soil and weather conditions.
    2.  **Fertilizer Recommendation:** Suggests the optimal fertilizer for your chosen crop, soil, and weather.
    Adjust the input values and select your desired recommendation type.
""")

st.markdown("---") # Visual separator

# --- Recommendation Mode Selection ---
st.sidebar.header("Choose Recommendation Type")
recommendation_mode = st.sidebar.radio(
    "Select Mode:",
    ("Crop Recommendation", "Fertilizer Recommendation"),
    index=0, # Default to Crop Recommendation
    help="Switch between recommending a crop or a fertilizer."
)

st.sidebar.markdown("---")
st.sidebar.info("Developed by Divyansh Sharma | [GitHub](https://github.com/oyedivyansh/AIML-Crop_fertlzr)")


# --- Input Section with Improved Layout ---
st.header("Input Your Farm Parameters")

# Using columns for a cleaner, side-by-side layout of input sliders
col1, col2 = st.columns(2)

# Collect common numerical inputs
with col1:
    st.subheader("Soil Conditions")
    soil_type_input = st.selectbox(
        "Select Soil Type:",
        options=sorted(unique_soil_types), # Sort options alphabetically for user-friendliness
        help="Choose the type of soil in your farm."
    )
    nitrogen = st.slider(
        "Nitrogen (N) Content (kg/ha):",
        min_value=0, max_value=200, value=80, step=1,
        help="Nitrogen is crucial for leaf growth and overall plant vigor."
    )
    phosphorus = st.slider(
        "Phosphorus (P) Content (kg/ha):",
        min_value=0, max_value=200, value=40, step=1,
        help="Phosphorus supports strong root development, flowering, and fruiting."
    )
    potassium = st.slider(
        "Potassium (K) Content (kg/ha):",
        min_value=0, max_value=200, value=40, step=1,
        help="Potassium is vital for overall plant health, disease resistance, and fruit quality."
    )


with col2:
    st.subheader("Environmental Factors")
    temperature = st.slider(
        "Temperature (°C):",
        min_value=0.0, max_value=50.0, value=25.0, step=0.1,
        help="Average air temperature in your cultivation area."
    )
    humidity = st.slider(
        "Humidity (%):",
        min_value=0.0, max_value=100.0, value=60.0, step=0.1,
        help="Relative humidity in the atmosphere."
    )
    moisture = st.slider(
        "Moisture (%):",
        min_value=0.0, max_value=100.0, value=50.0, step=0.1,
        help="Soil moisture content percentage."
    )
    # Rainfall is not in the new dataset, so removing this slider
    # rainfall = st.slider(
    #     "Rainfall (mm):",
    #     min_value=0.0, max_value=300.0, value=100.0, step=0.1,
    #     help="Total rainfall received in your area (e.g., average over the last month)."
    # )

# --- Conditional Input for Fertilizer Recommendation ---
selected_crop_for_fertilizer = None
if recommendation_mode == "Fertilizer Recommendation":
    st.subheader("Select Crop for Fertilizer Recommendation")
    selected_crop_for_fertilizer = st.selectbox(
        "Which crop are you growing?",
        options=sorted(unique_crop_types),
        help="This input is used specifically for fertilizer recommendation."
    )

st.markdown("---") # Another visual separator

# --- Prediction Button ---
if st.button("✨ Get Recommendation", type="primary", help="Click to receive the AI-powered recommendation."):
    try:
        # Encode Soil Type input
        encoded_soil_type = soil_type_le_for_crop_model.transform([soil_type_input])[0] # Use crop model's encoder for soil type as it's common

        # Prepare numerical inputs for both models (order matters based on feature_columns)
        # Note: Soil pH and Rainfall are removed, Moisture is added
        numerical_inputs = {
            'Temperature': temperature,
            'Humidity': humidity,
            'Moisture': moisture,
            'Nitrogen': nitrogen,
            'Potassium': potassium,
            'Phosphorus': phosphorus
        }

        if recommendation_mode == "Crop Recommendation":
            # Features for Crop Model: Temperature, Humidity, Moisture, Soil Type_encoded, Nitrogen, Potassium, Phosphorus
            crop_input_values = [
                numerical_inputs['Temperature'],
                numerical_inputs['Humidity'],
                numerical_inputs['Moisture'],
                encoded_soil_type, # Encoded Soil Type
                numerical_inputs['Nitrogen'],
                numerical_inputs['Potassium'],
                numerical_inputs['Phosphorus']
            ]
            # Ensure the order of features matches crop_feature_columns
            input_data_df = pd.DataFrame([crop_input_values], columns=crop_feature_columns)

            # Make prediction using the crop model
            prediction_encoded = crop_model.predict(input_data_df)[0]
            recommended_crop = crop_type_le_for_crop_model.inverse_transform([prediction_encoded])[0]

            st.success(f"**Recommended Crop:** {recommended_crop} 🌽")
            st.info("💡 **Important Note:** This recommendation is AI-generated. Always consider local climate, market demand, and agricultural practices.")

        elif recommendation_mode == "Fertilizer Recommendation":
            if selected_crop_for_fertilizer is None:
                st.warning("Please select a crop type for fertilizer recommendation.")
                st.stop()

            # Encode the selected crop type for the fertilizer model
            encoded_crop_type = crop_type_le_for_fert_model.transform([selected_crop_for_fertilizer])[0]

            # Prepare input for fertilizer model: Temperature, Humidity, Moisture, Soil Type_encoded, Crop Type_encoded, Nitrogen, Potassium, Phosphorus
            fertilizer_input_values = [
                numerical_inputs['Temperature'],
                numerical_inputs['Humidity'],
                numerical_inputs['Moisture'],
                encoded_soil_type, # Encoded Soil Type
                encoded_crop_type, # Encoded Crop Type
                numerical_inputs['Nitrogen'],
                numerical_inputs['Potassium'],
                numerical_inputs['Phosphorus']
            ]
            # Ensure the order of features matches fert_feature_columns
            input_data_df = pd.DataFrame([fertilizer_input_values], columns=fert_feature_columns)

            # Make prediction using the fertilizer model
            prediction_encoded = fert_model.predict(input_data_df)[0]
            recommended_fertilizer = fertilizer_le.inverse_transform([prediction_encoded])[0]

            st.success(f"**Recommended Fertilizer:** {recommended_fertilizer} 🧪")
            st.info("💡 **Important Note:** This recommendation is AI-generated. Always consult with local agricultural experts or perform a detailed soil test for precise, localized advice.")

        st.markdown("---")
        st.subheader("Your Input Summary:")
        # Display inputs in a neat format
        input_summary = {
            "Soil Type": soil_type_input,
            "Nitrogen (N)": f"{nitrogen} kg/ha",
            "Phosphorus (P)": f"{phosphorus} kg/ha",
            "Potassium (K)": f"{potassium} kg/ha",
            "Temperature": f"{temperature}°C",
            "Humidity": f"{humidity}%",
            "Moisture": f"{moisture}%"
        }
        if recommendation_mode == "Fertilizer Recommendation":
            input_summary["Selected Crop"] = selected_crop_for_fertilizer
        st.json(input_summary)

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.warning("Please ensure all input values are valid and the models are correctly trained with matching features. Check the console for more details.")

st.markdown("---")
st.markdown("Thank you for using the Smart Farm Recommendation System!")
