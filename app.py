from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from database import init_db, seed_database, insert_log, get_recent_logs

app = Flask(__name__)

# ===============================
# 1. Initialize Database
# ===============================
init_db()
seed_database()

# ===============================
# 2. Load Models & Encoders
# ===============================
try:
    scaler = joblib.load('scaler-2.pkl')
except FileNotFoundError:
    print("Warning: scaler-2.pkl not found, trying scaler.pkl")
    try:
         scaler = joblib.load('scaler.pkl')
    except Exception:
         scaler = None

try:
    encoders = joblib.load('encoders.pkl')
except Exception:
    encoders = None

MODELS = {}
model_names = {
    'rf': 'rf.pkl',
    'xgb': 'xgb.pkl',
    'svm': 'svm.pkl',
    'decision_tree': 'decision_tree.pkl',
    'logistic': 'logistic.pkl'
}

for friendly_name, pkl_file in model_names.items():
    try:
        MODELS[friendly_name] = joblib.load(pkl_file)
    except FileNotFoundError:
        print(f"Warning: Model file {pkl_file} not found. Skipping {friendly_name}.")
    except Exception as e:
        print(f"Warning: Failed to load {pkl_file} for {friendly_name}: {e}. Skipping.")

# The original columns order expected by the models (excluding Heart_Attack_Risk)
EXPECTED_COLUMNS = [
    "Age", "Gender", "Diabetes", "Hypertension",
    "Obesity", "Smoking", "Alcohol_Consumption", "Physical_Activity",
    "Diet_Score", "Cholesterol_Level", "Triglyceride_Level", "LDL_Level",
    "HDL_Level", "Systolic_BP", "Diastolic_BP", "Air_Pollution_Exposure",
    "Family_History", "Stress_Level", "Healthcare_Access",
    "Heart_Attack_History"
]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        model_choice = data.get('model', 'rf')
        
        if model_choice not in MODELS:
            return jsonify({'error': f'Model {model_choice} is not loaded.'}), 400
            
        model = MODELS[model_choice]

        # Extract values from form, provide sane defaults for others
        age = int(data.get('age', 40))
        gender = data.get('gender', 'Male')
        cholesterol = float(data.get('cholesterol', 200))
        systolic_bp = float(data.get('systolic', 120))
        diastolic_bp = float(data.get('diastolic', 80))
        diabetes = int(data.get('diabetes', 0))
        smoking = int(data.get('smoking', 0))
        heart_rate = float(data.get('heart_rate', 70)) # Not in ML, but used for logs
        patient_name = data.get('patient_name', 'Patient')
        
        hypertension = int(data.get('hypertension', 1 if systolic_bp > 130 else 0))
        obesity = int(data.get('obesity', 0))
        alcohol = float(data.get('alcohol', 0))
        physical_activity = float(data.get('physical_activity', 0))
        diet_score = float(data.get('diet_score', 5))
        triglyceride = float(data.get('triglyceride', 150))
        ldl = float(data.get('ldl', 100))
        hdl = float(data.get('hdl', 50))
        pollution = float(data.get('pollution', 0))
        family_history = int(data.get('family_history', 0))
        stress_level = float(data.get('stress_level', 5))
        healthcare = int(data.get('healthcare', 1))
        heart_attack_history = int(data.get('heart_attack_history', 0))

        # Construct dataframe for model prediction
        input_dict = {
            "Age": age,
            "Gender": gender,
            "Diabetes": diabetes,
            "Hypertension": hypertension,
            "Obesity": obesity,
            "Smoking": smoking,
            "Alcohol_Consumption": alcohol,
            "Physical_Activity": physical_activity,
            "Diet_Score": diet_score,
            "Cholesterol_Level": cholesterol,
            "Triglyceride_Level": triglyceride,
            "LDL_Level": ldl,
            "HDL_Level": hdl,
            "Systolic_BP": systolic_bp,
            "Diastolic_BP": diastolic_bp,
            "Air_Pollution_Exposure": pollution,
            "Family_History": family_history,
            "Stress_Level": stress_level,
            "Healthcare_Access": healthcare,
            "Heart_Attack_History": heart_attack_history
        }
        
        df_input = pd.DataFrame([input_dict], columns=EXPECTED_COLUMNS)
        
        # Apply label encodings safely
        if encoders:
            for col in ['State_Name', 'Gender']:
                if col in encoders and col in df_input.columns:
                    enc = encoders[col]
                    val = df_input[col].iloc[0]
                    # If unseen label, use the first class fallback
                    if val in enc.classes_:
                        df_input[col] = enc.transform([val])
                    else:
                        df_input[col] = enc.transform([enc.classes_[0]])

        # Some models used scaled data, some used raw data according to train.py
        # SVM and Logistic Regression used scaled data
        needs_scaling = ['svm', 'logistic']
        
        X = df_input
        if model_choice in needs_scaling and scaler is not None:
             X = scaler.transform(X)

        # Get probability of class 1
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)
            risk_prob = probs[0][1] * 100  # Convert to percentage
        elif hasattr(model, "decision_function"):
            decision = model.decision_function(X)
            # Approximate probability using sigmoid layer for SVM
            prob = 1.0 / (1.0 + np.exp(-decision[0]))
            risk_prob = prob * 100
        else:
            # Fallback to pure prediction
            pred = model.predict(X)
            risk_prob = pred[0] * 100
            
        # Convert numpy types to native python floats for JSON serialization
        risk_prob = float(risk_prob)
        
        # Save to logs
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        insert_log(patient_name, now_str, age, cholesterol, systolic_bp, diastolic_bp, heart_rate, model_choice, risk_prob)
        
        friendly_model_name = {
            'rf': 'Random Forest',
            'xgb': 'XGBoost',
            'svm': 'SVM',
            'decision_tree': 'Decision Tree',
            'logistic': 'Logistic Regression'
        }.get(model_choice, model_choice)

        return jsonify({
            'risk_percentage': round(risk_prob, 2),
            'model_used': friendly_model_name,
            'message': 'Prediction successful and recorded.'
        })

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/history', methods=['GET'])
def history():
    logs = get_recent_logs(20) # get up to 20
    # ensure ascending chronological order for chart
    logs = logs[::-1]
    return jsonify(logs)

if __name__ == '__main__':
    app.run(debug=True, port=8000)


