# ===============================
# 1. Install
# ===============================
# pip install xgboost joblib

# ===============================
# 2. Imports
# ===============================
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier

import joblib

# ===============================
# 3. Load Dataset
# ===============================
df = pd.read_csv('/content/heart_attack_prediction_india.csv')

# Clean column names (important)
df.columns = df.columns.str.strip()

# ===============================
# 4. Target
# ===============================
target = "Heart_Attack_Risk"

if target not in df.columns:
    raise ValueError("Target column not found. Check column names.")

X = df.drop(target, axis=1)
y = df[target]

# ===============================
# 5. Encode Categorical Columns
# ===============================
categorical_cols = X.select_dtypes(include=['object']).columns

encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le

# ===============================
# 6. Train-Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ===============================
# 7. Scaling
# ===============================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, "scaler.pkl")
joblib.dump(encoders, "encoders.pkl")

# ===============================
# 8. Logistic Regression
# ===============================
lr = LogisticRegression(max_iter=2000, class_weight='balanced')

lr_params = {
    'C': [0.01, 0.1, 1, 10]
}

lr_grid = GridSearchCV(lr, lr_params, cv=5, n_jobs=-1)
lr_grid.fit(X_train_scaled, y_train)

# ===============================
# 9. Random Forest
# ===============================
rf = RandomForestClassifier(class_weight='balanced')

rf_params = {
    'n_estimators': [200, 300, 500],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

rf_random = RandomizedSearchCV(
    rf, rf_params, n_iter=10, cv=5, n_jobs=-1, random_state=42
)
rf_random.fit(X_train, y_train)

# ===============================
# 10. Decision Tree
# ===============================
dt = DecisionTreeClassifier(class_weight='balanced')

dt_params = {
    'max_depth': [5, 10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}

dt_grid = GridSearchCV(dt, dt_params, cv=5, n_jobs=-1)
dt_grid.fit(X_train, y_train)

# ===============================
# 11. SVM
# ===============================
svm = SVC(class_weight='balanced')

svm_params = {
    'C': [1, 10],
    'kernel': ['rbf']
}

svm_grid = GridSearchCV(svm, svm_params, cv=5, n_jobs=-1)
svm_grid.fit(X_train_scaled, y_train)

# ===============================
# 12. XGBoost
# ===============================
xgb = XGBClassifier(eval_metric='logloss')

xgb_params = {
    'n_estimators': [200, 400],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8, 1.0]
}

xgb_random = RandomizedSearchCV(
    xgb, xgb_params, n_iter=10, cv=5, n_jobs=-1, random_state=42
)
xgb_random.fit(X_train, y_train)

# ===============================
# 13. Evaluation
# ===============================
models = {
    "Logistic Regression": (lr_grid.best_estimator_, X_test_scaled),
    "Random Forest": (rf_random.best_estimator_, X_test),
    "Decision Tree": (dt_grid.best_estimator_, X_test),
    "SVM": (svm_grid.best_estimator_, X_test_scaled),
    "XGBoost": (xgb_random.best_estimator_, X_test)
}

best_model = None
best_score = 0

for name, (model, X_data) in models.items():
    y_pred = model.predict(X_data)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n{name}")
    print("Accuracy:", acc)
    print(classification_report(y_test, y_pred))

    if acc > best_score:
        best_score = acc
        best_model = name

print(f"\n🔥 Best Model: {best_model} with accuracy {best_score}")

# ===============================
# 14. Save ALL Models
# ===============================
joblib.dump(lr_grid.best_estimator_, "logistic.pkl")
joblib.dump(rf_random.best_estimator_, "rf.pkl")
joblib.dump(dt_grid.best_estimator_, "decision_tree.pkl")
joblib.dump(svm_grid.best_estimator_, "svm.pkl")
joblib.dump(xgb_random.best_estimator_, "xgb.pkl")

print("\nAll models saved successfully.")