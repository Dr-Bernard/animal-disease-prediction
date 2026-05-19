# app.py
from flask import Flask, request, render_template_string
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load saved objects
model = joblib.load('disease_model.pkl')
scaler = joblib.load('scaler.pkl')
le = joblib.load('label_encoder.pkl')
feature_cols = joblib.load('feature_columns.pkl')

# HTML form template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Animal Disease Predictor</title>
    <style>
        body { font-family: Arial; margin: 40px; }
        input, select { margin: 5px; padding: 5px; }
        .container { max-width: 500px; margin: auto; }
    </style>
</head>
<body>
    <div class="container">
        <h2>🐄 Livestock Disease Prediction</h2>
        <form method="post">
            <label>Animal type:</label>
            <select name="animal_type">
                <option value="Cow">Cow</option>
                <option value="Dog">Dog</option>
                <option value="Goat">Goat</option>
                <option value="Horse">Horse</option>
                <option value="Pig">Pig</option>
                <option value="Rabbit">Rabbit</option>
                <option value="Sheep">Sheep</option>
            </select><br>
            <label>Appetite loss (1=yes,0=no):</label> <input type="number" name="appetite" step="1"><br>
            <label>Vomiting (1=yes,0=no):</label> <input type="number" name="vomiting" step="1"><br>
            <label>Diarrhea (1=yes,0=no):</label> <input type="number" name="diarrhea" step="1"><br>
            <label>Coughing (1=yes,0=no):</label> <input type="number" name="coughing" step="1"><br>
            <label>Labored breathing (1=yes,0=no):</label> <input type="number" name="labored" step="1"><br>
            <label>Lameness (1=yes,0=no):</label> <input type="number" name="lameness" step="1"><br>
            <label>Skin lesions (1=yes,0=no):</label> <input type="number" name="skin" step="1"><br>
            <label>Nasal discharge (1=yes,0=no):</label> <input type="number" name="nasal" step="1"><br>
            <label>Body temperature (°C):</label> <input type="number" name="temp" step="any"><br>
            <label>Heart rate (bpm):</label> <input type="number" name="heart" step="any"><br>
            <input type="submit" value="Predict">
        </form>
        {% if prediction %}
            <h3>Predicted disease: {{ prediction }}</h3>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def predict():
    prediction = None
    if request.method == 'POST':
        # Get form data
        animal = request.form['animal_type']
        appetite = int(request.form['appetite'])
        vomiting = int(request.form['vomiting'])
        diarrhea = int(request.form['diarrhea'])
        coughing = int(request.form['coughing'])
        labored = int(request.form['labored'])
        lameness = int(request.form['lameness'])
        skin = int(request.form['skin'])
        nasal = int(request.form['nasal'])
        temp = float(request.form['temp'])
        heart = float(request.form['heart'])
        fever = 1 if temp > 39.5 else 0

        # Build data dictionary
        data = {
            'Appetite_Loss': appetite,
            'Vomiting': vomiting,
            'Diarrhea': diarrhea,
            'Coughing': coughing,
            'Labored_Breathing': labored,
            'Lameness': lameness,
            'Skin_Lesions': skin,
            'Nasal_Discharge': nasal,
            'Body_Temperature': temp,
            'Heart_Rate': heart,
            'Fever': fever
        }
        # One‑hot encode animal type
        animal_cols = [col for col in feature_cols if col.startswith('Animal_Type_')]
        for col in animal_cols:
            data[col] = 0
        selected_col = f'Animal_Type_{animal}'
        if selected_col in data:
            data[selected_col] = 1

        df = pd.DataFrame([data])
        df = df[feature_cols]
        scaled = scaler.transform(df)
        pred_num = model.predict(scaled)[0]
        prediction = le.inverse_transform([pred_num])[0]

    return render_template_string(HTML_TEMPLATE, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)