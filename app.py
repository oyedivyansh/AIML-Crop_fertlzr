# train_fertilizer_model.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# Load the dataset
df = pd.read_csv("Fertilizer Prediction.csv")

# Select only relevant columns: Crop, N, P, K, and the target
df = df[["Crop", "N", "P", "K", "Fertilizer Name"]]

# One-hot encode Crop
df = pd.get_dummies(df, columns=["Crop"])

# Features and Target
X = df.drop("Fertilizer Name", axis=1)
y = df["Fertilizer Name"]

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model and scaler
joblib.dump(model, "fertilizer_model.sav")
joblib.dump(scaler, "scaler.sav")
