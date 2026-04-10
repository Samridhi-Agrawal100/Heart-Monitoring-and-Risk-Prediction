from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np
from PIL import Image
import pickle
from pathlib import Path
from io import BytesIO
from datetime import datetime
from database import init_db, seed_database, insert_log, get_recent_logs

try:
    import torch
    import torch.nn as nn
    import timm
    from torchvision import transforms
    ECG_IMPORT_ERROR = None
except Exception as exc:
    torch = None
    nn = None
    timm = None
    transforms = None
    ECG_IMPORT_ERROR = str(exc)

app = Flask(__name__)

# ===============================
# 1. Initialize Database
# ===============================
init_db()
seed_database()

# ===============================
# 2. Load Models & Encoders
# ===============================
BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIRS = [
    BASE_DIR,
    BASE_DIR / 'heart_attack_model',
    BASE_DIR / 'CAD_model',
    BASE_DIR / 'ECG_model',
    BASE_DIR / 'ecg_model',
    BASE_DIR / 'models',
    BASE_DIR / 'model'
]


def resolve_artifact(filename, search_dirs=None):
    dirs = search_dirs if search_dirs is not None else ARTIFACT_DIRS
    for folder in dirs:
        candidate = folder / filename
        if candidate.exists():
            return candidate
    return None


def load_first_available(candidates, label, required=True, search_dirs=None):
    for filename in candidates:
        artifact_path = resolve_artifact(filename, search_dirs=search_dirs)
        if artifact_path is None:
            continue
        try:
            return joblib.load(artifact_path)
        except Exception as exc:
            print(f"Warning: Failed to load {filename} from {artifact_path}: {exc}")
            return None

    if required:
        print(f"Warning: {label} not found in artifact directories.")
    return None


scaler = load_first_available(['scaler-2.pkl', 'scaler.pkl'], 'Scaler', required=False)
encoders = load_first_available(['encoders.pkl', 'encoder.pkl'], 'Encoders')

MODELS = {}
model_names = {
    'rf': 'rf.pkl',
    'xgb': 'xgb.pkl'
}

for friendly_name, pkl_file in model_names.items():
    artifact_path = resolve_artifact(pkl_file)
    if artifact_path is None:
        print(f"Warning: Model file {pkl_file} not found. Skipping {friendly_name}.")
        continue

    try:
        MODELS[friendly_name] = joblib.load(artifact_path)
    except Exception as e:
        print(f"Warning: Failed to load {pkl_file} from {artifact_path} for {friendly_name}: {e}. Skipping.")


CAD_DIRS = [BASE_DIR / 'CAD_model', BASE_DIR]
CAD_MODEL = load_first_available(['cardio_model.pkl'], 'CAD model', required=False, search_dirs=CAD_DIRS)
CAD_SCALER = load_first_available(['scaler.pkl'], 'CAD scaler', required=False, search_dirs=CAD_DIRS)
CAD_FEATURES = load_first_available(['features.pkl'], 'CAD features', required=False, search_dirs=CAD_DIRS)

ECG_MODEL = load_first_available(
    ['ecg_model.pkl', 'ecg_classifier.pkl', 'ecg.pkl', 'model.pkl'],
    'ECG model',
    required=False
)
ECG_KERAS_PATH = resolve_artifact('ecg_model.h5') or resolve_artifact('ecg_model.keras')
ECG_KERAS_MODEL = None
ECG_PTH_PATH = resolve_artifact('ecg_model.pth')
ECG_LABELS_PATH = resolve_artifact('labels.pkl')


if nn is not None:
    class MultiAxisAttention(nn.Module):
        def __init__(self, channels, reduction_ratio=16):
            super().__init__()
            reduced_channels = channels // reduction_ratio
            self.mlp = nn.Sequential(
                nn.Conv2d(channels, reduced_channels, 1, bias=False),
                nn.ReLU(),
                nn.Conv2d(reduced_channels, channels, 1, bias=False),
            )
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x_w = torch.mean(x, dim=2, keepdim=True)
            w = self.sigmoid(self.mlp(x_w))
            x = x * w
            x_h = torch.mean(x, dim=3, keepdim=True)
            h = self.sigmoid(self.mlp(x_h))
            return x * h


    class CustomMaxViT(nn.Module):
        def __init__(self, backbone, channels, num_classes):
            super().__init__()
            self.backbone = backbone
            self.ma_attn = MultiAxisAttention(channels)
            self.norm = nn.LayerNorm(channels)
            self.head = nn.Linear(channels, num_classes)

        def forward(self, x):
            x = self.backbone.forward_features(x)
            x = self.ma_attn(x)
            x = x.mean(dim=(-2, -1))
            x = self.norm(x)
            return self.head(x)


def load_ecg_labels(path):
    if path is None:
        return {}

    with open(path, 'rb') as handle:
        label_to_id = pickle.load(handle)

    if isinstance(label_to_id, dict):
        return {v: k for k, v in label_to_id.items()}

    if isinstance(label_to_id, (list, tuple)):
        return {index: str(value) for index, value in enumerate(label_to_id)}

    return {}


def load_ecg_pytorch_model():
    if ECG_IMPORT_ERROR is not None:
        return None, None, None, f"Missing ECG dependencies: {ECG_IMPORT_ERROR}. Install torch, torchvision, and timm."

    if ECG_PTH_PATH is None:
        return None, None, None, 'ecg_model.pth not found in project directories.'

    try:
        backbone = timm.create_model(
            'maxvit_base_tf_384.in1k',
            pretrained=False,
            num_classes=0,
            global_pool=''
        )

        model = CustomMaxViT(backbone, 768, 4)
        model.load_state_dict(torch.load(ECG_PTH_PATH, map_location='cpu'))
        model.eval()

        label_map = load_ecg_labels(ECG_LABELS_PATH)
        transform_pipeline = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        return model, label_map, transform_pipeline, None
    except Exception as exc:
        return None, None, None, f"Failed to load ECG PyTorch model: {exc}"


ECG_PYTORCH_MODEL, ECG_ID_TO_LABEL, ECG_TRANSFORM, ECG_LOAD_ERROR = load_ecg_pytorch_model()


def model_probability(model, features):
    """Return the probability of the positive class as a percentage."""
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(features)[0][1] * 100)

    if hasattr(model, "decision_function"):
        decision = model.decision_function(features)
        probability = 1.0 / (1.0 + np.exp(-decision[0]))
        return float(probability * 100)

    prediction = model.predict(features)
    return float(prediction[0] * 100)


def normalize_gender(value):
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {'male', 'm', '1'}:
            return 1
        if text in {'female', 'f', '0'}:
            return 0
    return int(value)


def normalize_level_1_to_3(value, mild_threshold, severe_threshold):
    numeric = float(value)
    if numeric in {1.0, 2.0, 3.0}:
        return int(numeric)
    if numeric < mild_threshold:
        return 1
    if numeric <= severe_threshold:
        return 2
    return 3


def predict_cardio(input_data):
    if CAD_MODEL is None or CAD_SCALER is None or CAD_FEATURES is None:
        raise ValueError('CAD artifacts missing. Required: cardio_model.pkl, scaler.pkl, features.pkl in CAD_model folder.')

    df = pd.DataFrame([input_data])

    # Feature engineering required by CAD model.
    df['BMI'] = df['weight'] / ((df['height'] / 100.0) ** 2)
    df['pulse_pressure'] = df['ap_hi'] - df['ap_lo']
    if float(df['ap_lo'].iloc[0]) == 0:
        raise ValueError('ap_lo cannot be 0 for CAD prediction.')

    df['bp_ratio'] = df['ap_hi'] / df['ap_lo']

    df = df[CAD_FEATURES]
    df_scaled = CAD_SCALER.transform(df)

    pred = int(CAD_MODEL.predict(df_scaled)[0])
    prob = float(CAD_MODEL.predict_proba(df_scaled)[0][1]) if hasattr(CAD_MODEL, 'predict_proba') else None

    return pred, prob


def ecg_label_name(class_index):
    if isinstance(ECG_ID_TO_LABEL, dict):
        return str(ECG_ID_TO_LABEL.get(class_index, class_index))
    return str(class_index)


def predict_ecg_from_image(image_bytes):
    if ECG_PYTORCH_MODEL is None:
        if ECG_LOAD_ERROR:
            raise ValueError(ECG_LOAD_ERROR)
        raise ValueError('ECG model not loaded. Ensure ecg_model.pth and labels.pkl are available.')

    image = Image.open(BytesIO(image_bytes)).convert('RGB')
    tensor = ECG_TRANSFORM(image).unsqueeze(0)

    with torch.no_grad():
        outputs = ECG_PYTORCH_MODEL(tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred_class = torch.max(probs, 1)

    class_index = int(pred_class.item())
    conf = float(confidence.item())

    return {
        'class_index': class_index,
        'label': ecg_label_name(class_index),
        'probability': round(conf * 100, 2),
        'preprocess': 'rgb_384x384_imagenet_norm'
    }

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
        model_choice = data.get('model', 'ensemble')
        
        if model_choice != 'ensemble' and model_choice not in MODELS:
            return jsonify({'error': f'Model {model_choice} is not loaded.'}), 400

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

        X = df_input

        rf_prob = None
        xgb_prob = None

        if model_choice == 'ensemble':
            missing_models = [name for name in ('rf', 'xgb') if name not in MODELS]
            if missing_models:
                return jsonify({'error': f"Ensemble requires RF and XGB, but missing: {', '.join(missing_models)}"}), 400

            rf_prob = model_probability(MODELS['rf'], X)
            xgb_prob = model_probability(MODELS['xgb'], X)
            risk_prob = (rf_prob + xgb_prob) / 2.0
            model_label = 'Ensemble (RF + XGBoost)'
        else:
            risk_prob = model_probability(MODELS[model_choice], X)
            model_label = {
                'rf': 'Random Forest',
                'xgb': 'XGBoost'
            }.get(model_choice, model_choice)

        risk_prob = float(risk_prob)
        
        # Save to logs
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        insert_log(patient_name, now_str, age, cholesterol, systolic_bp, diastolic_bp, heart_rate, model_label, risk_prob)

        return jsonify({
            'risk_percentage': round(risk_prob, 2),
            'model_used': model_label,
            'ensemble_breakdown': {
                'rf': round(rf_prob, 2) if rf_prob is not None else None,
                'xgb': round(xgb_prob, 2) if xgb_prob is not None else None,
                'method': 'average' if model_choice == 'ensemble' else 'single-model'
            },
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


@app.route('/predict/cad', methods=['POST'])
def predict_cad_route():
    try:
        data = request.json or {}
        required_fields = [
            'age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo',
            'cholesterol', 'gluc', 'smoke', 'alco', 'active'
        ]
        missing = [field for field in required_fields if field not in data]
        if missing:
            return jsonify({'error': f"Missing CAD fields: {', '.join(missing)}"}), 400

        input_data = {
            'age': float(data['age']),
            'gender': normalize_gender(data['gender']),
            'height': float(data['height']),
            'weight': float(data['weight']),
            'ap_hi': float(data['ap_hi']),
            'ap_lo': float(data['ap_lo']),
            'cholesterol': normalize_level_1_to_3(data['cholesterol'], mild_threshold=200, severe_threshold=240),
            'gluc': normalize_level_1_to_3(data['gluc'], mild_threshold=100, severe_threshold=126),
            'smoke': int(data['smoke']),
            'alco': int(data['alco']),
            'active': int(data['active'])
        }

        pred, prob = predict_cardio(input_data)

        return jsonify({
            'prediction': pred,
            'risk_label': 'High CAD Risk' if pred == 1 else 'Lower CAD Risk',
            'probability': round(prob * 100, 2) if prob is not None else None,
            'message': 'CAD prediction successful.'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict/ecg', methods=['POST'])
def predict_ecg_route():
    try:
        image_file = request.files.get('ecg_image') or request.files.get('file')
        if image_file is None:
            return jsonify({'error': 'No ECG image file provided. Use field name ecg_image or file.'}), 400

        if image_file.filename == '':
            return jsonify({'error': 'Empty ECG image filename.'}), 400

        image_bytes = image_file.read()
        if not image_bytes:
            return jsonify({'error': 'Uploaded ECG image is empty.'}), 400

        result = predict_ecg_from_image(image_bytes)

        return jsonify({
            'prediction': result['class_index'],
            'label': result['label'],
            'probability': result['probability'],
            'preprocess': result['preprocess'],
            'message': 'ECG prediction completed.'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=8000)


