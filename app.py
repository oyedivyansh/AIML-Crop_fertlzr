import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Set page config with custom styling
st.set_page_config(
    page_title="🌱 Smart Agriculture Assistant",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .main-header {
        background: linear-gradient(90deg, #4CAF50, #45a049);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 4px solid #4CAF50;
    }
    
    .recommendation-box {
        background: linear-gradient(135deg, #4CAF50, #45a049);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .input-section {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    .stTab {
        background: white;
        border-radius: 10px;
        padding: 1rem;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #4CAF50, #45a049);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: bold;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(76, 175, 80, 0.4);
    }
    
    .sidebar .stSelectbox {
        background: white;
        border-radius: 10px;
    }
    
    .info-card {
        background: #e8f5e8;
        border: 1px solid #4CAF50;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .warning-card {
        background: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load models and scalers with error handling
@st.cache_resource
def load_models():
    try:
        crop_model = joblib.load("crop_recmd_model.sav")
        crop_scaler = joblib.load("crop_recmnd_scale.sav")
        fertilizer_model = joblib.load("fertilizer_model.sav")
        fertilizer_scaler = joblib.load("scaler.sav")
        return crop_model, crop_scaler, fertilizer_model, fertilizer_scaler
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        return None, None, None, None

# Load fertilizer dataset
@st.cache_data
def load_fertilizer_data():
    try:
        fert_df = pd.read_csv("fertilizer.csv")
        if "Crop" not in fert_df.columns:
            st.error("❌ \'Crop\' column not found in fertilizer.csv")
            return None
        return fert_df
    except Exception as e:
        st.error(f"❌ Error loading fertilizer data: {e}")
        return None

# Load data
crop_model, crop_scaler, fertilizer_model, fertilizer_scaler = load_models()
fert_df = load_fertilizer_data()

# Crop labels for crop recommendation model
crop_labels = [
    "Rice", "Maize", "Chickpea", "Kidney Beans", "Pigeon Peas", "Moth Beans",
    "Mung Bean", "Black Gram", "Lentil", "Pomegranate", "Banana", "Mango",
    "Grapes", "Watermelon", "Muskmelon", "Apple", "Orange", "Papaya",
    "Coconut", "Cotton", "Jute", "Coffee"
]

# Header
st.markdown("""
<div class="main-header">
    <h1>🌱 Smart Agriculture Assistant</h1>
    <p style="font-size: 1.2rem; margin-top: 1rem;">
        Get intelligent recommendations for crops and fertilizers based on soil conditions and environmental factors
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar with information
with st.sidebar:
    st.markdown("### 📊 About This Tool")
    st.markdown("""
    <div class="info-card">
        <h4>🎯 Purpose</h4>
        <p>This AI-powered tool helps farmers and agricultural professionals make informed decisions about:</p>
        <ul>
            <li>🌾 Optimal crop selection</li>
            <li>💊 Fertilizer recommendations</li>
            <li>📈 Yield optimization</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="warning-card">
        <h4>⚠️ Important Note</h4>
        <p>These recommendations are based on machine learning models. Always consult with local agricultural experts for best results.</p>
    </div>
    """, unsafe_allow_html=True)

# Main content tabs
tab1, tab2 = st.tabs(["🌾 Crop Recommendation", "💊 Fertilizer Recommendation"])

# --- Crop Recommendation Tab ---
with tab1:
    st.markdown("### 🌾 Find the Best Crop for Your Soil")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="input-section">
            <h4>🧪 Soil Nutrients</h4>
        </div>
        """, unsafe_allow_html=True)
        
        N = st.slider("🔵 Nitrogen (N)", 0, 140, 50, help="Nitrogen content in soil (mg/kg)")
        P = st.slider("🟡 Phosphorus (P)", 5, 145, 50, help="Phosphorus content in soil (mg/kg)")
        K = st.slider("🔴 Potassium (K)", 5, 205, 50, help="Potassium content in soil (mg/kg)")
        ph = st.number_input("⚖️ pH Level", 0.0, 14.0, 6.5, step=0.1, help="Soil pH level (0-14)")
    
    with col2:
        st.markdown("""
        <div class="input-section">
            <h4>🌤️ Environmental Conditions</h4>
        </div>
        """, unsafe_allow_html=True)
        
        temperature = st.number_input("🌡️ Temperature (°C)", 0.0, 60.0, 25.0, step=0.5, help="Average temperature")
        humidity = st.number_input("💧 Humidity (%)", 0.0, 100.0, 50.0, step=1.0, help="Relative humidity percentage")
        rainfall = st.number_input("🌧️ Rainfall (mm)", 0.0, 500.0, 100.0, step=5.0, help="Annual rainfall in millimeters")

    # Display current conditions summary
    st.markdown("### 📋 Current Conditions Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🧪 NPK Ratio", f"{N}-{P}-{K}")
    with col2:
        st.metric("🌡️ Temperature", f"{temperature}°C")
    with col3:
        st.metric("💧 Humidity", f"{humidity}%")
    with col4:
        st.metric("🌧️ Rainfall", f"{rainfall}mm")

    if st.button("🔍 Get Crop Recommendation", key="crop_btn"):
        if crop_model and crop_scaler:
            with st.spinner("🤖 Analyzing soil conditions..."):
                input_data = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]],
                                          columns=["N", "P", "K", "temperature", "humidity", "ph", "rainfall"])
                scaled_input = crop_scaler.transform(input_data)
                prediction = crop_model.predict(scaled_input)

                try:
                    crop_name = crop_labels[int(prediction[0])]
                    st.markdown(f"""
                    <div class="recommendation-box">
                        <h2>🎯 Recommended Crop</h2>
                        <h1 style="margin: 1rem 0; font-size: 3rem;">{crop_name}</h1>
                        <p style="font-size: 1.2rem;">Based on your soil and environmental conditions, this crop has the highest probability of success!</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Additional information
                    st.markdown("### 💡 Why This Crop?")
                    st.info(f"The AI model analyzed your soil nutrients (N: {N}, P: {P}, K: {K}), environmental conditions (Temp: {temperature}°C, Humidity: {humidity}%, Rainfall: {rainfall}mm), and pH level ({ph}) to determine that **{crop_name}** is the most suitable crop for your conditions.")
                    
                except Exception as e:
                    st.error(f"🚨 Could not interpret prediction: {e}")
        else:
            st.error("❌ Models not loaded properly. Please check if model files exist.")

# --- Fertilizer Recommendation Tab ---
with tab2:
    st.markdown("### 💊 Get Optimal Fertilizer Recommendation")
    
    if fert_df is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="input-section">
                <h4>🌱 Crop Selection</h4>
            </div>
            """, unsafe_allow_html=True)
            
            crop_options = sorted(fert_df["Crop"].dropna().unique())
            selected_crop = st.selectbox("🌾 Select Your Crop", crop_options, help="Choose the crop you\'re planning to grow")
            
            st.markdown("""
            <div class="input-section">
                <h4>🧪 Current Soil Nutrients</h4>
            </div>
            """, unsafe_allow_html=True)
            
            N_fert = st.slider("🔵 Nitrogen Level (N)", 0, 140, 50, key="fn", help="Current nitrogen level in soil")
            P_fert = st.slider("🟡 Phosphorus Level (P)", 5, 145, 50, key="fp", help="Current phosphorus level in soil")
            K_fert = st.slider("🔴 Potassium Level (K)", 5, 205, 50, key="fk", help="Current potassium level in soil")
        
        with col2:
            st.markdown("""
            <div class="input-section">
                <h4>🌤️ Environmental Conditions</h4>
            </div>
            """, unsafe_allow_html=True)
            
            temperature_fert = st.number_input("🌡️ Temperature (°C)", 0.0, 60.0, 25.0, key="ftemp", step=0.5)
            humidity_fert = st.number_input("💧 Humidity (%)", 0.0, 100.0, 50.0, key="fhum", step=1.0)
            moisture = st.number_input("🌊 Soil Moisture (%)", 0.0, 100.0, 30.0, key="fmoisture", step=1.0, help="Current soil moisture content")

        # Display fertilizer conditions summary
        st.markdown("### 📋 Fertilizer Analysis Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🌾 Selected Crop", selected_crop)
        with col2:
            st.metric("🧪 NPK Levels", f"{N_fert}-{P_fert}-{K_fert}")
        with col3:
            st.metric("🌡️ Conditions", f"{temperature_fert}°C, {humidity_fert}%")
        with col4:
            st.metric("🌊 Soil Moisture", f"{moisture}%")

        if st.button("🔍 Get Fertilizer Recommendation", key="fert_btn"):
            if fertilizer_model and fertilizer_scaler:
                with st.spinner("🧬 Analyzing fertilizer requirements..."):
                    try:
                        # Prepare input dataframe
                        input_df = pd.DataFrame([[selected_crop, N_fert, P_fert, K_fert, temperature_fert, humidity_fert, moisture]],
                                                columns=["Crop Type", "Nitrogen", "Phosphorous", "Potassium", "Temperature", "Humidity", "Moisture"])

                        # One-hot encode the Crop Type column
                        fert_df_crop_dummies = pd.get_dummies(fert_df["Crop Type"])
                        input_encoded = pd.get_dummies(input_df, columns=["Crop Type"])

                        # Align input_encoded columns with training columns
                        expected_columns = list(fert_df_crop_dummies.columns) + ["Nitrogen", "Phosphorous", "Potassium", "Temperature", "Humidity", "Moisture"]
                        input_encoded = input_encoded.reindex(columns=expected_columns, fill_value=0)

                        # Scale the input
                        scaled_input = fertilizer_scaler.transform(input_encoded)

                        # Predict fertilizer recommendation
                        prediction = fertilizer_model.predict(scaled_input)

                        st.markdown(f"""
                        <div class="recommendation-box">
                            <h2>🎯 Recommended Fertilizer</h2>
                            <h1 style="margin: 1rem 0; font-size: 3rem;">{prediction[0]}</h1>
                            <p style="font-size: 1.2rem;">This fertilizer is optimally suited for your {selected_crop} crop under current conditions!</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Additional fertilizer information
                        st.markdown("### 💡 Fertilizer Application Guide")
                        st.info(f"For **{selected_crop}** with current NPK levels ({N_fert}-{P_fert}-{K_fert}) and environmental conditions (Temperature: {temperature_fert}°C, Humidity: {humidity_fert}%, Soil Moisture: {moisture}%), **{prediction[0]}** fertilizer will provide the optimal nutrient balance for healthy growth and maximum yield.")
                        
                        # Best practices
                        st.markdown("### 🌟 Application Best Practices")
                        st.markdown("""
                        - 📅 Apply fertilizer during the recommended growth stage
                        - 🌧️ Consider weather conditions before application
                        - 💧 Ensure adequate soil moisture for nutrient absorption
                        - 📏 Follow recommended dosage guidelines
                        - 🔄 Monitor plant response and adjust if necessary
                        """)

                    except Exception as e:
                        st.error(f"🚨 Error during fertilizer prediction: {e}")
            else:
                st.error("❌ Fertilizer models not loaded properly. Please check if model files exist.")
    else:
        st.error("❌ Fertilizer data not loaded properly. Please check if fertilizer.csv exists.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>🌱 Smart Agriculture Assistant | Powered by Machine Learning</p>
    <p>Made with ❤️ for sustainable farming</p>
</div>
""", unsafe_allow_html=True)
