import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# 1. Load data
df = pd.read_csv('cleaned_animal_disease_prediction.csv')
print("Original shape:", df.shape)

# 2. Clean binary columns
binary_cols = ['Appetite_Loss', 'Vomiting', 'Diarrhea', 'Coughing', 
               'Labored_Breathing', 'Lameness', 'Skin_Lesions', 
               'Nasal_Discharge', 'Eye_Discharge']
for col in binary_cols:
    df[col] = df[col].map({'Yes': 1, 'No': 0})

# 3. Clean temperature and add Fever feature
df['Body_Temperature'] = df['Body_Temperature'].str.replace('°C', '').astype(float)
df['Fever'] = (df['Body_Temperature'] > 39.5).astype(int)

# 4. Group rare diseases (keep those with >=10 occurrences)
disease_counts = df['Disease_Prediction'].value_counts()
common_diseases = disease_counts[disease_counts >= 10].index
df['Disease_Prediction'] = df['Disease_Prediction'].apply(
    lambda x: x if x in common_diseases else 'Other'
)
print(f"Number of disease classes after grouping: {df['Disease_Prediction'].nunique()}")

# 5. Select features (dropped Weight and Eye_Discharge for simplicity)
feature_cols = [
    'Animal_Type', 'Appetite_Loss', 'Vomiting', 'Diarrhea', 'Coughing',
    'Labored_Breathing', 'Lameness', 'Skin_Lesions', 'Nasal_Discharge',
    'Body_Temperature', 'Heart_Rate', 'Fever'
]
target = 'Disease_Prediction'
df = df[feature_cols + [target]].dropna()

# 6. One-hot encode Animal_Type
df = pd.get_dummies(df, columns=['Animal_Type'], drop_first=True)

# 7. Encode target
le = LabelEncoder()
y = le.fit_transform(df[target])
X = df.drop(target, axis=1)
print("Feature columns:", X.columns.tolist())

# 8. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 9. Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 10. Train with balanced class weights
model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model.fit(X_train_scaled, y_train)

# 11. Evaluate on test set
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest accuracy: {accuracy:.2%}")

# Classification report for classes present in test set
present_labels = np.unique(y_test)
present_names = le.classes_[present_labels]
print("\nClassification Report (test set):")
print(classification_report(y_test, y_pred, labels=present_labels, target_names=present_names))

# 12. Example predictions
print("\nExample predictions (first 10 test samples):")
for i in range(10):
    true_name = le.inverse_transform([y_test[i]])[0]
    pred_name = le.inverse_transform([y_pred[i]])[0]
    print(f"True: {true_name:30} Pred: {pred_name}")

# 13. Cross‑validation on full dataset (scale entire X)
X_scaled_full = scaler.fit_transform(X)   # scale all data
model_cv = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
cv_scores = cross_val_score(model_cv, X_scaled_full, y, cv=5)
print(f"\nCross‑validation accuracy: {cv_scores.mean():.2%} (+/- {cv_scores.std():.2%})")


import joblib

# Save model, scaler, label encoder, and feature columns
joblib.dump(model, 'disease_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(le, 'label_encoder.pkl')
joblib.dump(X.columns.tolist(), 'feature_columns.pkl')

print("✅ Model and preprocessors saved successfully!")