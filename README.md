# Animal Disease Prediction for Nigerian Livestock

Machine learning prototype to predict diseases (FMD, CBPP, PPR, Trypanosomiasis, Healthy) in cattle, goats, and sheep based on vital signs and symptoms.

## Features
- Trained on 200 synthetic records with 100% accuracy (ideal scenario)
- CLI and web interface for predictions

## How to run
1. Clone the repo
2. Create virtual environment: `python -m venv venv`
3. Activate it: `venv\Scripts\activate` (Windows)
4. Install requirements: `pip install -r requirements.txt`
5. Run training: `python disease_prediction.py`
6. Try CLI: `python predict_cli.py`
7. Launch web app: `python app.py` and open http://localhost:5000