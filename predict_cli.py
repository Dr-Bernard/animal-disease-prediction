# predict_cli.py
import joblib
import pandas as pd
import numpy as np

# Load saved objects
model = joblib.load('disease_model.pkl')
scaler = joblib.load('scaler.pkl')
le = joblib.load('label_encoder.pkl')
feature_cols = joblib.load('feature_columns.pkl')

print("\n🐄 Animal Disease Prediction CLI")
print("Enter the animal's information:\n")

# Get user input
animal_type = input("Animal type (Cow/Dog/Goat/Horse/Pig/Rabbit/Sheep): ").strip().capitalize()
appetite = int(input("Appetite loss? (1=yes, 0=no): "))
vomiting = int(input("Vomiting? (1=yes, 0=no): "))
diarrhea = int(input("Diarrhea? (1=yes, 0=no): "))
coughing = int(input("Coughing? (1=yes, 0=no): "))
labored_breathing = int(input("Labored breathing? (1=yes, 0=no): "))
lameness = int(input("Lameness? (1=yes, 0=no): "))
skin_lesions = int(input("Skin lesions? (1=yes, 0=no): "))
nasal_discharge = int(input("Nasal discharge? (1=yes, 0=no): "))
temperature = float(input("Body temperature (°C): "))
heart_rate = float(input("Heart rate (bpm): "))

# Compute fever (temperature > 39.5)
fever = 1 if temperature > 39.5 else 0

# Build a dictionary with all features
data = {
    'Appetite_Loss': appetite,
    'Vomiting': vomiting,
    'Diarrhea': diarrhea,
    'Coughing': coughing,
    'Labored_Breathing': labored_breathing,
    'Lameness': lameness,
    'Skin_Lesions': skin_lesions,
    'Nasal_Discharge': nasal_discharge,
    'Body_Temperature': temperature,
    'Heart_Rate': heart_rate,
    'Fever': fever
}

# Add one‑hot encoded animal type columns (all 0 except the chosen one)
animal_cols = [col for col in feature_cols if col.startswith('Animal_Type_')]
for col in animal_cols:
    data[col] = 0
# Set the selected animal to 1
selected_col = f'Animal_Type_{animal_type}'
if selected_col in data:
    data[selected_col] = 1
else:
    print(f"Warning: '{animal_type}' is not a known animal type. Using default (all zeros).")

# Create DataFrame, ensure column order matches training
df = pd.DataFrame([data])
df = df[feature_cols]

# Scale and predict
scaled = scaler.transform(df)
pred_num = model.predict(scaled)[0]
disease = le.inverse_transform([pred_num])[0]

print(f"\n🔮 Predicted disease: {disease}\n")