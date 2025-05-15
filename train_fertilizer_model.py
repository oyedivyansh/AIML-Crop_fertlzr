import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# Load dataset
df = pd.read_csv("Fertilizer Prediction.csv")

# Keep only relevant columns
df = df[["Crop", "N", "P", "K", "Fertilizer Name"]]

# One-hot encode Crop
df = pd.get_dummies(df, columns=["Crop"])

# Split features and target
X = df.drop("Fertilizer Name", axis=1)
y = df["Fertilizer Name"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = RandomForestClassifier()
model.fit(X_scaled, y)

# Save model and scaler
joblib.dump(model, "fertilizer_model.sav")
joblib.dump(scaler, "scaler.sav")
