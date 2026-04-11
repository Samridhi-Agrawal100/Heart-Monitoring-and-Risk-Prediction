from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np
from PIL import Image
import pickle
from pathlib import Path
from io import BytesIO
from datetime import datetime
from database import init_db, insert_prediction, get_recent_logs
from openai import OpenAI
import json
import os

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


def load_env_file(env_path):
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding='utf-8').splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue

        key, value = line.split('=', 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


load_env_file(Path(__file__).resolve().parent / '.env')

GROQ_API_KEY = os.environ.get('GROQ_API_KEY', '').strip()
GROQ_MODEL = os.environ.get('GROQ_MODEL', 'llama-3.1-70b-versatile').strip()

# ===============================
# 1. Initialize Database
# ===============================
init_db()

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
        label_map = load_ecg_labels(ECG_LABELS_PATH)
        checkpoint = torch.load(ECG_PTH_PATH, map_location='cpu')

        # Support checkpoints saved as raw state_dict or wrapped dicts.
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        if not isinstance(state_dict, dict):
            return None, None, None, 'Invalid ECG checkpoint format: expected a state_dict.'

        # Remove DataParallel prefix if present.
        cleaned_state_dict = {}
        for key, value in state_dict.items():
            new_key = key[7:] if key.startswith('module.') else key
            cleaned_state_dict[new_key] = value

        backbone = timm.create_model(
            'maxvit_base_tf_384.in1k',
            pretrained=False,
            num_classes=0,
            global_pool=''
        )

        channels = getattr(backbone, 'num_features', 768)
        num_classes = len(label_map) if label_map else 4
        if 'head.weight' in cleaned_state_dict and hasattr(cleaned_state_dict['head.weight'], 'shape'):
            num_classes = int(cleaned_state_dict['head.weight'].shape[0])

        model = CustomMaxViT(backbone, channels, num_classes)
        missing, unexpected = model.load_state_dict(cleaned_state_dict, strict=False)
        if unexpected:
            print(f"Warning: Unexpected ECG checkpoint keys: {unexpected}")
        if missing:
            print(f"Warning: Missing ECG checkpoint keys: {missing}")

        model.eval()

        if not label_map:
            label_map = {index: str(index) for index in range(num_classes)}

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
        proba = model.predict_proba(features)
        if len(proba.shape) != 2 or proba.shape[1] == 0:
            raise ValueError("Model predict_proba output has unexpected shape.")

        class_index = 1 if proba.shape[1] > 1 else 0
        if hasattr(model, "classes_") and len(model.classes_) == proba.shape[1]:
            classes = list(model.classes_)
            for preferred_positive in (1, "1", True):
                if preferred_positive in classes:
                    class_index = classes.index(preferred_positive)
                    break

        return float(proba[0][class_index] * 100)

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


def reverse_binary_flag(value):
    return 0 if int(value) else 1


def parse_groq_explanation(raw_text):
    """Parse Groq response into summary and recommendations."""
    lines = raw_text.strip().split('\n')
    summary_lines = []
    recommendations = []
    
    for line in lines:
        line = line.strip()
        # Check if it's a numbered recommendation (e.g., "1. ...", "2. ...")
        if line and line[0].isdigit() and '.' in line[:3]:
            # Remove number, dot, and asterisks from the line
            clean = line.split('.', 1)[1].strip() if '.' in line else line
            clean = clean.replace('**', '').replace('*', '').strip()
            if clean:
                recommendations.append(clean)
        elif line and not any(line.startswith(str(i) + '.') for i in range(1, 10)):
            # This is part of the summary
            summary_lines.append(line.replace('**', '').replace('*', '').strip())
    
    # First 2-3 sentences for summary
    summary = ' '.join(summary_lines[:100]).strip()  # Take first part, limit length
    summary = ' '.join(summary.split())
    
    return {
        'summary': summary[:300],  # Keep summary reasonably short
        'recommendations': recommendations[:5],  # Max 5 recommendations
    }


def build_cad_context_for_llm(input_data, pred, prob):
    bmi = input_data['weight'] / ((input_data['height'] / 100.0) ** 2) if input_data['height'] else None
    pulse_pressure = input_data['ap_hi'] - input_data['ap_lo']
    bp_ratio = input_data['ap_hi'] / input_data['ap_lo'] if input_data['ap_lo'] else None

    smoking_status = 'Yes' if int(input_data.get('smoke_raw', input_data.get('smoke', 0))) else 'No'
    alcohol_status = 'Yes' if int(input_data.get('alco_raw', input_data.get('alco', 0))) else 'No'

    return {
        'age': input_data['age'],
        'gender': 'Male' if input_data['gender'] == 1 else 'Female',
        'height': input_data['height'],
        'weight': input_data['weight'],
        'bmi': round(bmi, 2) if bmi is not None else None,
        'ap_hi': input_data['ap_hi'],
        'ap_lo': input_data['ap_lo'],
        'pulse_pressure': round(pulse_pressure, 2),
        'bp_ratio': round(bp_ratio, 2) if bp_ratio is not None else None,
        'cholesterol': input_data['cholesterol'],
        'cholesterol_raw': input_data.get('cholesterol_raw'),
        'gluc': input_data['gluc'],
        'gluc_raw': input_data.get('gluc_raw'),
        'smoke': input_data['smoke'],
        'alco': input_data['alco'],
        'smoking_status': smoking_status,
        'alcohol_status': alcohol_status,
        'active': input_data['active'],
        'prediction': int(pred),
        'probability': round(float(prob) * 100, 2) if prob is not None else None,
        'target_bp': '<120/80 mmHg',
        'target_cholesterol': '<200 mg/dL (model level 1)',
        'target_glucose': '<100 mg/dL fasting (model level 1)',
        'target_activity': 'at least 150 min/week moderate activity',
    }


def build_cad_recommendations(context):
    recommendations = []

    if context.get('smoking_status') == 'Yes':
        recommendations.append('Stop smoking to reduce CAD risk.')

    if context.get('alcohol_status') == 'Yes':
        recommendations.append('Limit alcohol use to minimize its negative impact.')

    recommendations.extend([
        f'Keep blood pressure near {context.get("target_bp")} (current {context.get("ap_hi")}/{context.get("ap_lo")}).',
        f'Keep cholesterol near {context.get("target_cholesterol")} (current model level {context.get("cholesterol")}).',
        f'Keep fasting glucose near {context.get("target_glucose")} (current model level {context.get("gluc")}).',
        f'Maintain physical activity at {context.get("target_activity")}.',
        'Follow a lower-salt, lower-saturated-fat eating pattern daily.',
    ])

    return recommendations[:5]


def explain_cad_prediction_with_groq(input_data, pred, prob):
    context = build_cad_context_for_llm(input_data, pred, prob)
    fallback = (
        f"CAD model predicted {'High CAD Risk' if pred == 1 else 'Lower CAD Risk'} with "
        f"{context['probability']}% confidence. Key inputs: age {context['age']}, BP {context['ap_hi']}/{context['ap_lo']}, "
        f"cholesterol {context['cholesterol']}, glucose {context['gluc']}, smoking {context['smoke']}, alcohol {context['alco']}, activity {context['active']}."
    )

    if not GROQ_API_KEY:
        return {
            'provider': 'local-fallback',
            'summary': fallback,
            'recommendations': build_cad_recommendations(context),
        }

    try:
        client = OpenAI(
            api_key=GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1",
        )
        
        prompt = (
            'You are a clinical explainer for a CAD risk dashboard. Use only the patient metrics below. '
            'Write 1 clear plain-language summary paragraph and exactly 5 numbered actions. '
            'Keep the total response around 150-210 words. '
            'Do not use markdown bullets, bold text, headings, or extra commentary. '
            'Do not say "consult a doctor" unless absolutely necessary. '
            'Use the actual lifestyle flags exactly as written. '
            'Smoking history and alcohol use are the real user inputs, not model features. '
            'Only mention smoking cessation if Smoking history is Yes. '
            'Only mention alcohol reduction if Alcohol Use is Yes. '
            'If those flags are No, omit them entirely and do not comment on them. '
            'If physical activity is Yes, say to maintain it and include the target range. '
            'Whenever you mention blood pressure, cholesterol, or glucose, include safe target ranges and compare with current values. '
            'Focus on concrete, patient-specific guidance using the values provided.\n'
            f'Age: {context.get("age")}\n'
            f'Blood Pressure: {context.get("ap_hi")}/{context.get("ap_lo")}\n'
            f'Cholesterol Level (model): {context.get("cholesterol")}\n'
            f'Cholesterol raw: {context.get("cholesterol_raw")} mg/dL\n'
            f'Glucose (model): {context.get("gluc")}\n'
            f'Glucose raw: {context.get("gluc_raw")} mg/dL\n'
            f'Smoking history: {context.get("smoking_status")}\n'
            f'Alcohol use: {context.get("alcohol_status")}\n'
            f'Physical Activity: {"Yes" if context.get("active") else "No"}\n'
            f'Target BP: {context.get("target_bp")}\n'
            f'Target Cholesterol: {context.get("target_cholesterol")}\n'
            f'Target Glucose: {context.get("target_glucose")}\n'
            f'Target Activity: {context.get("target_activity")}\n'
            f'Risk Probability: {context.get("probability")}%'
        )
        
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {'role': 'system', 'content': 'You are a direct clinical advisor. Use the specific patient data provided. Be concrete, readable, and avoid generic advice.'},
                {'role': 'user', 'content': prompt},
            ],
            temperature=0.45,
            max_tokens=300,
        )
        
        content = response.choices[0].message.content.strip()
        parsed = parse_groq_explanation(content)
        return {
            'provider': 'groq',
            'summary': parsed['summary'],
            'recommendations': parsed['recommendations'],
        }
    except Exception as exc:
        return {
            'provider': 'local-fallback',
            'summary': f'{fallback} Groq explanation could not be loaded: {exc}',
            'recommendations': build_cad_recommendations(context),
        }


def build_heart_attack_context_for_llm(input_data, pred, prob):
    """Build context dict for heart attack LLM explanation."""
    return {
        'age': input_data.get('Age'),
        'gender': input_data.get('Gender', 'Unknown'),
        'cholesterol': input_data.get('Cholesterol_Level'),
        'systolic_bp': input_data.get('Systolic_BP'),
        'diastolic_bp': input_data.get('Diastolic_BP'),
        'triglyceride': input_data.get('Triglyceride_Level'),
        'ldl': input_data.get('LDL_Level'),
        'hdl': input_data.get('HDL_Level'),
        'diet_score': input_data.get('Diet_Score'),
        'stress_level': input_data.get('Stress_Level'),
        'pollution': input_data.get('Air_Pollution_Exposure'),
        'physical_activity': input_data.get('Physical_Activity'),
        'smoking': input_data.get('Smoking'),
        'alcohol': input_data.get('Alcohol_Consumption'),
        'diabetes': input_data.get('Diabetes'),
        'hypertension': input_data.get('Hypertension'),
        'obesity': input_data.get('Obesity'),
        'family_history': input_data.get('Family_History'),
        'heart_attack_history': input_data.get('Heart_Attack_History'),
        'prediction': int(pred),
        'probability': round(float(prob) * 100, 2) if prob is not None else None,
    }


def explain_heart_attack_prediction_with_groq(input_data, pred, prob):
    """Explain heart attack prediction using Groq LLM."""
    context = build_heart_attack_context_for_llm(input_data, pred, prob)
    fallback = (
        f"Heart attack risk model predicted {'High Risk' if pred == 1 else 'Lower Risk'} with "
        f"{context['probability']}% confidence based on age {context['age']}, BP {context['systolic_bp']}/{context['diastolic_bp']}, "
        f"cholesterol {context['cholesterol']}, smoking {context['smoking']}, and other cardiovascular factors."
    )

    if not GROQ_API_KEY:
        return {
            'provider': 'local-fallback',
            'summary': fallback,
            'recommendations': [
                'Monitor blood pressure regularly and maintain a healthy diet.',
                'Increase physical activity to at least 150 minutes per week.',
                'Avoid smoking and limit alcohol consumption.',
                'Manage stress through relaxation and lifestyle changes.',
                'Schedule regular check-ups with your healthcare provider.',
            ],
        }

    try:
        client = OpenAI(
            api_key=GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1",
        )
        
        prompt = (
            'You are a clinical AI analyzing heart attack risk. '
            'IMPORTANT: Use ONLY the specific health metrics provided. Do NOT make generic recommendations or ask the patient to "consult a doctor" for general advice. '
            'Here are the specific metrics from this patient:\n'
            f'- Age: {context.get("age")}\n'
            f'- Cholesterol: {context.get("cholesterol")} mg/dL\n'
            f'- Blood Pressure: {context.get("systolic_bp")}/{context.get("diastolic_bp")}\n'
            f'- LDL: {context.get("ldl")}, HDL: {context.get("hdl")}, Triglycerides: {context.get("triglyceride")}\n'
            f'- Smoking: {"Yes" if context.get("smoking") else "No"}\n'
            f'- Diabetes: {"Yes" if context.get("diabetes") else "No"}\n'
            f'- Physical Activity: {"Yes" if context.get("physical_activity") else "No"}\n'
            f'- Stress Level: {context.get("stress_level")}/10\n'
            f'- Risk Probability: {context.get("probability")}%\n\n'
            'Based on THESE ACTUAL VALUES (not generic patient profiles):\n'
            '1. Explain briefly why the model predicted this risk level USING THE SPECIFIC METRICS ABOVE\n'
            '2. List 3-5 SPECIFIC actions to reduce risk, tailored to THIS patient\'s actual risk factors\n'
            '3. Be concrete - mention specific values and what needs to change\n'
            'Example: "Your cholesterol at 240 is high - aim to lower to <200" NOT "manage cholesterol"\n'
            'Example: "You smoke - stop smoking within the next month" NOT "consult doctor about smoking"\n'
            'Do not say "consult a doctor" unless absolutely necessary. Give actionable steps.'
        )
        
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {'role': 'system', 'content': 'You are a direct clinical advisor. Use the specific patient data provided. Never give generic "consult a doctor" advice when you can be specific. Be actionable and concrete.'},
                {'role': 'user', 'content': prompt},
            ],
            temperature=0.6,
            max_tokens=300,
        )
        
        content = response.choices[0].message.content.strip()
        parsed = parse_groq_explanation(content)
        return {
            'provider': 'groq',
            'summary': parsed['summary'],
            'recommendations': parsed['recommendations'],
        }
    except Exception as exc:
        return {
            'provider': 'local-fallback',
            'summary': f'{fallback} Groq explanation could not be loaded: {exc}',
            'recommendations': [
                'Monitor blood pressure regularly and maintain a healthy diet.',
                'Increase physical activity to at least 150 minutes per week.',
                'Avoid smoking and limit alcohol consumption.',
                'Manage stress through relaxation and lifestyle changes.',
                'Schedule regular check-ups with your healthcare provider.',
            ],
        }


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
    prob = None
    if hasattr(CAD_MODEL, 'predict_proba'):
        proba = CAD_MODEL.predict_proba(df_scaled)
        if len(proba.shape) != 2 or proba.shape[1] == 0:
            raise ValueError('CAD model predict_proba output has unexpected shape.')

        class_index = 1 if proba.shape[1] > 1 else 0
        if hasattr(CAD_MODEL, 'classes_') and len(CAD_MODEL.classes_) == proba.shape[1]:
            classes = list(CAD_MODEL.classes_)
            for preferred_positive in (1, '1', True):
                if preferred_positive in classes:
                    class_index = classes.index(preferred_positive)
                    break

        prob = float(proba[0][class_index])

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


@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')


@app.route('/cad')
def cad_page():
    return render_template('cad.html')


@app.route('/ecg')
def ecg_page():
    return render_template('ecg.html')


@app.route('/health-history')
def health_history_page():
    return render_template('health_history.html')

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

        # RF/XGB were trained on unscaled features in train.py, so keep raw encoded values.
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
        
        # Get Groq explanation for the prediction
        explanation = explain_heart_attack_prediction_with_groq(input_dict, 1 if risk_prob >= 50 else 0, risk_prob / 100.0)
        
        # Save heart attack prediction logs with vitals.
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        insert_prediction(
            {
                'patient_name': patient_name,
                'date': now_str,
                'prediction_type': 'heart_attack',
                'model_used': model_label,
                'risk_percentage': risk_prob,
                'risk_label': 'High Risk' if risk_prob >= 65 else ('Moderate Risk' if risk_prob >= 35 else 'Lower Risk'),
                'age': age,
                'gender': gender,
                'heart_rate': heart_rate,
                'cholesterol_level': cholesterol,
                'systolic_bp': systolic_bp,
                'diastolic_bp': diastolic_bp,
                'triglyceride_level': triglyceride,
                'ldl_level': ldl,
                'hdl_level': hdl,
                'glucose_level': data.get('gluc', None),
                'stress_level': stress_level,
                'pollution_exposure': pollution,
                'physical_activity': physical_activity,
            }
        )

        return jsonify({
            'risk_percentage': round(risk_prob, 2),
            'model_used': model_label,
            'explanation': explanation,
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
    logs = get_recent_logs(20, prediction_type='heart_attack') # get up to 20
    # ensure ascending chronological order for chart
    logs = logs[::-1]
    return jsonify(logs)


@app.route('/history/full', methods=['GET'])
def history_full():
    logs = get_recent_logs(1000, prediction_type=None)
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
            'cholesterol_raw': float(data['cholesterol']),
            'gluc_raw': float(data['gluc']),
            'cholesterol': normalize_level_1_to_3(data['cholesterol'], mild_threshold=200, severe_threshold=240),
            'gluc': normalize_level_1_to_3(data['gluc'], mild_threshold=100, severe_threshold=126),
            'smoke_raw': int(data['smoke']),
            'alco_raw': int(data['alco']),
            'smoke': reverse_binary_flag(data['smoke']),
            'alco': reverse_binary_flag(data['alco']),
            'active': int(data['active'])
        }

        pred, prob = predict_cardio(input_data)
        explanation = explain_cad_prediction_with_groq(input_data, pred, prob)

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        risk_score = float(prob * 100) if prob is not None else (100.0 if pred == 1 else 0.0)
        insert_prediction(
            {
                'patient_name': 'Patient',
                'date': now_str,
                'prediction_type': 'cad',
                'model_used': 'CAD Model',
                'risk_percentage': risk_score,
                'risk_label': 'High CAD Risk' if pred == 1 else 'Lower CAD Risk',
                'age': input_data['age'],
                'gender': str(input_data['gender']),
                'cholesterol_level': input_data['cholesterol'],
                'systolic_bp': input_data['ap_hi'],
                'diastolic_bp': input_data['ap_lo'],
                'glucose_level': input_data['gluc'],
                'physical_activity': input_data['active'],
            }
        )

        return jsonify({
            'prediction': pred,
            'risk_label': 'High CAD Risk' if pred == 1 else 'Lower CAD Risk',
            'probability': round(prob * 100, 2) if prob is not None else None,
            'explanation': explanation,
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

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        insert_prediction(
            {
                'patient_name': 'Patient',
                'date': now_str,
                'prediction_type': 'ecg',
                'model_used': 'ECG Model',
                'risk_percentage': result['probability'],
                'risk_label': result['label'],
                'extra_json': result['preprocess'],
            }
        )

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


