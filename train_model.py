# train_models.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle
import os

# --- Configuration for file paths ---
DATA_FILE = 'Fertilizer_Prediction.csv' # Updated file name
DATA_PATH = os.path.join('data', DATA_FILE)

MODEL_DIR = 'models'
CROP_MODEL_FILE = 'crop_model.pkl'
FERTILIZER_MODEL_FILE = 'fertilizer_model.pkl'

CROP_MODEL_PATH = os.path.join(MODEL_DIR, CROP_MODEL_FILE)
FERTILIZER_MODEL_PATH = os.path.join(MODEL_DIR, FERTILIZER_MODEL_FILE)

# Ensure the 'models' directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

print(f"Starting model training process for both Crop and Fertilizer Recommendation...")
print(f"Attempting to load data from: {DATA_PATH}")

# --- 1. Data Loading ---
try:
    df = pd.read_csv(DATA_PATH)
    print(f"Successfully loaded {len(df)} rows from {DATA_FILE}.")
except FileNotFoundError:
    print(f"Error: The data file '{DATA_FILE}' was not found.")
    print(f"Please ensure '{DATA_FILE}' is placed inside the 'data/' directory.")
    exit(1)
except Exception as e:
    print(f"An unexpected error occurred while loading data: {e}")
    exit(1)

# --- Pre-processing: Correcting column names from the provided data ---
df.rename(columns={
    'Temparature': 'Temperature',
    'Phosphorous': 'Phosphorus'
}, inplace=True)
print("Corrected column names: 'Temparature' to 'Temperature', 'Phosphorous' to 'Phosphorus'.")


# --- 2. Data Preprocessing (Common for both models) ---
print("Starting data preprocessing steps...")

# Define common numerical features based on the provided data
numerical_cols = ['Temperature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorus']
# Define categorical features that need encoding
categorical_cols_to_encode = ['Soil Type', 'Crop Type', 'Fertilizer Name']

# Check if all required columns exist in the loaded DataFrame
required_cols_for_both = numerical_cols + categorical_cols_to_encode
missing_cols = [col for col in required_cols_for_both if col not in df.columns]
if missing_cols:
    print(f"Error: Your CSV is missing the following required columns: {missing_cols}")
    print("Please ensure your '{DATA_FILE}' has all the necessary columns for both crop and fertilizer predictions.")
    exit(1)

# Handle potential missing values in numerical features (using median imputation)
for col in numerical_cols:
    if df[col].isnull().any():
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        print(f"Filled missing values in '{col}' with median: {median_val}")

# Initialize a dictionary to store all label encoders
label_encoders = {}

# Encode 'Soil Type'
le_soil_type = LabelEncoder()
df['Soil Type_encoded'] = le_soil_type.fit_transform(df['Soil Type'])
label_encoders['Soil Type'] = le_soil_type
print(f"Encoded 'Soil Type'. Unique values mapped: {le_soil_type.classes_}")

# Encode 'Crop Type'
le_crop = LabelEncoder()
df['Crop Type_encoded'] = le_crop.fit_transform(df['Crop Type'])
label_encoders['Crop Type'] = le_crop
print(f"Encoded 'Crop Type'. Unique values mapped: {le_crop.classes_}")

# Encode 'Fertilizer Name'
le_fertilizer = LabelEncoder()
df['Fertilizer Name_encoded'] = le_fertilizer.fit_transform(df['Fertilizer Name'])
label_encoders['Fertilizer Name'] = le_fertilizer
print(f"Encoded 'Fertilizer Name'. Unique fertilizers mapped: {le_fertilizer.classes_}")

print("Data preprocessing complete.")

# --- 3. Model Training - Crop Recommendation Model ---
print("\n--- Training Crop Recommendation Model ---")
# Features for Crop Recommendation: Temperature, Humidity, Moisture, Soil Type_encoded, Nitrogen, Potassium, Phosphorus
# Target: Crop Type_encoded
crop_input_features = ['Temperature', 'Humidity', 'Moisture', 'Soil Type_encoded', 'Nitrogen', 'Potassium', 'Phosphorus']
crop_features_df = df[crop_input_features]
crop_target_encoded = df['Crop Type_encoded']

# Store the exact list of feature column names for the crop model
crop_model_feature_columns = crop_features_df.columns.tolist()

X_crop_train, X_crop_test, y_crop_train, y_crop_test = train_test_split(
    crop_features_df, crop_target_encoded, test_size=0.2, random_state=42
)

crop_model = RandomForestClassifier(n_estimators=100, random_state=42)
crop_model.fit(X_crop_train, y_crop_train)
print("Crop Recommendation Model training finished.")
crop_accuracy = crop_model.score(X_crop_test, y_crop_test)
print(f"Crop Recommendation Model accuracy on test set: {crop_accuracy:.2f}")

# Save Crop Model and Associated Data
# This model needs 'Crop Type' and 'Soil Type' encoders for decoding/encoding inputs
crop_model_data_to_save = {
    'model': crop_model,
    'label_encoders': {'Crop Type': le_crop, 'Soil Type': le_soil_type},
    'feature_columns': crop_model_feature_columns
}
with open(CROP_MODEL_PATH, 'wb') as file:
    pickle.dump(crop_model_data_to_save, file)
print(f"Crop model saved to: {CROP_MODEL_PATH}")


# --- 4. Model Training - Fertilizer Recommendation Model ---
print("\n--- Training Fertilizer Recommendation Model ---")
# Features for Fertilizer Recommendation: Temperature, Humidity, Moisture, Soil Type_encoded, Crop Type_encoded, Nitrogen, Potassium, Phosphorus
# Target: Fertilizer Name_encoded
fertilizer_input_features = ['Temperature', 'Humidity', 'Moisture', 'Soil Type_encoded', 'Crop Type_encoded', 'Nitrogen', 'Potassium', 'Phosphorus']
fertilizer_features_df = df[fertilizer_input_features]
fertilizer_target_encoded = df['Fertilizer Name_encoded']

# Store the exact list of feature column names for the fertilizer model
fertilizer_model_feature_columns = fertilizer_features_df.columns.tolist()

X_fert_train, X_fert_test, y_fert_train, y_fert_test = train_test_split(
    fertilizer_features_df, fertilizer_target_encoded, test_size=0.2, random_state=42
)

fertilizer_model = RandomForestClassifier(n_estimators=100, random_state=42)
fertilizer_model.fit(X_fert_train, y_fert_train)
print("Fertilizer Recommendation Model training finished.")
fertilizer_accuracy = fertilizer_model.score(X_fert_test, y_fert_test)
print(f"Fertilizer Recommendation Model accuracy on test set: {fertilizer_accuracy:.2f}")

# Save Fertilizer Model and Associated Data
# This model needs 'Fertilizer Name', 'Crop Type', and 'Soil Type' encoders
fertilizer_model_data_to_save = {
    'model': fertilizer_model,
    'label_encoders': {'Fertilizer Name': le_fertilizer, 'Crop Type': le_crop, 'Soil Type': le_soil_type},
    'feature_columns': fertilizer_model_feature_columns
}
with open(FERTILIZER_MODEL_PATH, 'wb') as file:
    pickle.dump(fertilizer_model_data_to_save, file)
print(f"Fertilizer model saved to: {FERTILIZER_MODEL_PATH}")

print("\nAll models trained and saved successfully!")
print("Please ensure 'crop_model.pkl' and 'fertilizer_model.pkl' are in the 'models/' directory before running the Streamlit app.")
