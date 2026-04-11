import sqlite3
import random
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

DB_NAME = "database.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # Table to store heart attack prediction logs
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS health_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_name TEXT,
        date TEXT,
        age INTEGER,
        cholesterol_level REAL,
        systolic_bp REAL,
        diastolic_bp REAL,
        heart_rate REAL, 
        model_used TEXT,
        risk_percentage REAL
    )
    """)

    # New unified table used by current Flask app.
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS prediction_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_name TEXT,
            date TEXT NOT NULL,
            prediction_type TEXT NOT NULL,
            model_used TEXT,
            risk_percentage REAL,
            risk_label TEXT,
            age REAL,
            gender TEXT,
            heart_rate REAL,
            cholesterol_level REAL,
            systolic_bp REAL,
            diastolic_bp REAL,
            triglyceride_level REAL,
            ldl_level REAL,
            hdl_level REAL,
            glucose_level REAL,
            stress_level REAL,
            pollution_exposure REAL,
            physical_activity REAL,
            bmi REAL,
            extra_json TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def insert_prediction(record: Dict[str, Any]) -> None:
    """Unified insert used by app.py for heart attack, CAD, and ECG predictions."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO prediction_logs (
            patient_name, date, prediction_type, model_used, risk_percentage, risk_label,
            age, gender, heart_rate, cholesterol_level, systolic_bp, diastolic_bp,
            triglyceride_level, ldl_level, hdl_level, glucose_level, stress_level,
            pollution_exposure, physical_activity, bmi, extra_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            record.get("patient_name"),
            record.get("date"),
            record.get("prediction_type"),
            record.get("model_used"),
            record.get("risk_percentage"),
            record.get("risk_label"),
            record.get("age"),
            record.get("gender"),
            record.get("heart_rate"),
            record.get("cholesterol_level"),
            record.get("systolic_bp"),
            record.get("diastolic_bp"),
            record.get("triglyceride_level"),
            record.get("ldl_level"),
            record.get("hdl_level"),
            record.get("glucose_level"),
            record.get("stress_level"),
            record.get("pollution_exposure"),
            record.get("physical_activity"),
            record.get("bmi"),
            record.get("extra_json"),
        ),
    )

    # Keep legacy table in sync for heart attack history compatibility.
    if record.get("prediction_type") == "heart_attack":
        cursor.execute(
            """
            INSERT INTO health_logs (
                patient_name, date, age, cholesterol_level, systolic_bp,
                diastolic_bp, heart_rate, model_used, risk_percentage
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.get("patient_name"),
                record.get("date"),
                record.get("age"),
                record.get("cholesterol_level"),
                record.get("systolic_bp"),
                record.get("diastolic_bp"),
                record.get("heart_rate"),
                record.get("model_used"),
                record.get("risk_percentage"),
            ),
        )

    conn.commit()
    conn.close()

def insert_log(patient_name, date, age, cholesterol_level, systolic_bp, diastolic_bp, heart_rate, model_used, risk_percentage):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO health_logs (patient_name, date, age, cholesterol_level, systolic_bp, diastolic_bp, heart_rate, model_used, risk_percentage)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (patient_name, date, age, cholesterol_level, systolic_bp, diastolic_bp, heart_rate, model_used, risk_percentage))
    conn.commit()
    conn.close()

def get_recent_logs(limit=10):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    # Fetch recent logs ordered by date
    cursor.execute("""
    SELECT date, risk_percentage, cholesterol_level, systolic_bp, diastolic_bp, heart_rate, model_used 
    FROM health_logs 
    ORDER BY date DESC LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()
    
    # format rows into dict
    logs = []
    for r in rows:
        logs.append({
            "date": r[0],
            "risk_percentage": r[1],
            "cholesterol_level": r[2],
            "systolic_bp": r[3],
            "diastolic_bp": r[4],
            "heart_rate": r[5],
            "model_used": r[6]
        })
    return logs


def get_recent_logs(limit: int = 10, prediction_type: Optional[str] = "heart_attack") -> List[Dict[str, Any]]:
    """Return recent logs in the schema expected by the current frontend/app.

    Falls back to legacy health_logs table if prediction_logs is empty.
    """
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Prefer unified table used by current app.
    if prediction_type:
        cursor.execute(
            """
            SELECT *
            FROM prediction_logs
            WHERE prediction_type = ?
            ORDER BY date DESC
            LIMIT ?
            """,
            (prediction_type, limit),
        )
    else:
        cursor.execute(
            """
            SELECT *
            FROM prediction_logs
            ORDER BY date DESC
            LIMIT ?
            """,
            (limit,),
        )

    rows = cursor.fetchall()
    if rows:
        conn.close()
        return [
            {
                "date": row["date"],
                "risk_percentage": row["risk_percentage"],
                "cholesterol_level": row["cholesterol_level"],
                "systolic_bp": row["systolic_bp"],
                "diastolic_bp": row["diastolic_bp"],
                "heart_rate": row["heart_rate"],
                "model_used": row["model_used"],
                "prediction_type": row["prediction_type"],
                "risk_label": row["risk_label"],
                "age": row["age"],
                "gender": row["gender"],
                "triglyceride_level": row["triglyceride_level"],
                "ldl_level": row["ldl_level"],
                "hdl_level": row["hdl_level"],
                "glucose_level": row["glucose_level"],
                "stress_level": row["stress_level"],
                "pollution_exposure": row["pollution_exposure"],
                "physical_activity": row["physical_activity"],
            }
            for row in rows
        ]

    # Fallback for old table so old data is still visible.
    cursor.execute(
        """
        SELECT date, risk_percentage, cholesterol_level, systolic_bp,
               diastolic_bp, heart_rate, model_used, age
        FROM health_logs
        ORDER BY date DESC
        LIMIT ?
        """,
        (limit,),
    )
    legacy_rows = cursor.fetchall()
    conn.close()

    return [
        {
            "date": row["date"],
            "risk_percentage": row["risk_percentage"],
            "cholesterol_level": row["cholesterol_level"],
            "systolic_bp": row["systolic_bp"],
            "diastolic_bp": row["diastolic_bp"],
            "heart_rate": row["heart_rate"],
            "model_used": row["model_used"],
            "prediction_type": "heart_attack",
            "risk_label": None,
            "age": row["age"],
            "gender": None,
            "triglyceride_level": None,
            "ldl_level": None,
            "hdl_level": None,
            "glucose_level": None,
            "stress_level": None,
            "pollution_exposure": None,
            "physical_activity": None,
        }
        for row in legacy_rows
    ]

def seed_database():
    """Seed the database with 10 days of past data showing risk decreasing."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM health_logs")
    if cursor.fetchone()[0] == 0:
        base_date = datetime.now() - timedelta(days=10)
        cholesterol = 280
        sys_bp = 160
        dia_bp = 100
        hr = 85
        risk = 75.0
        
        for i in range(10):
            date_str = (base_date + timedelta(days=i)).strftime("%Y-%m-%d %H:%M:%S")
            insert_log("Siddh", date_str, 45, cholesterol, sys_bp, dia_bp, hr, "Ensemble (RF + XGBoost)", round(risk, 2))
            
            # Simulate health improvement
            cholesterol -= random.uniform(2, 6)
            sys_bp -= random.uniform(1, 3)
            dia_bp -= random.uniform(0.5, 2)
            hr -= random.uniform(0.5, 1.5)
            risk -= random.uniform(2, 5)
            
            if risk < 5: risk = 5
            
    conn.close()

if __name__ == "__main__":
    init_db()
    seed_database()
    print("Database initialized and seeded.")
