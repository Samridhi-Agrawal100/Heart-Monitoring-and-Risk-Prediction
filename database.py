import sqlite3
from typing import Any, Dict, List, Optional

DB_NAME = "database.db"


def _get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = _get_connection()
    cursor = conn.cursor()

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


def reset_all_data() -> None:
    """Delete all logged prediction records so the app can start fresh."""
    conn = _get_connection()
    cursor = conn.cursor()

    cursor.execute("DELETE FROM prediction_logs")
    cursor.execute("DELETE FROM sqlite_sequence WHERE name = 'prediction_logs'")

    # Clean legacy table if it still exists from older versions.
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='health_logs'"
    )
    if cursor.fetchone() is not None:
        cursor.execute("DELETE FROM health_logs")
        cursor.execute("DELETE FROM sqlite_sequence WHERE name = 'health_logs'")

    conn.commit()
    conn.close()


def insert_prediction(record: Dict[str, Any]) -> None:
    conn = _get_connection()
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

    conn.commit()
    conn.close()


def insert_log(
    patient_name: str,
    date: str,
    age: float,
    cholesterol_level: float,
    systolic_bp: float,
    diastolic_bp: float,
    heart_rate: float,
    model_used: str,
    risk_percentage: float,
) -> None:
    """Backward-compatible helper for heart attack predictions."""
    insert_prediction(
        {
            "patient_name": patient_name,
            "date": date,
            "prediction_type": "heart_attack",
            "model_used": model_used,
            "risk_percentage": risk_percentage,
            "age": age,
            "heart_rate": heart_rate,
            "cholesterol_level": cholesterol_level,
            "systolic_bp": systolic_bp,
            "diastolic_bp": diastolic_bp,
        }
    )


def _row_to_log(row: sqlite3.Row) -> Dict[str, Any]:
    return {
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


def get_recent_logs(limit: int = 10, prediction_type: Optional[str] = "heart_attack") -> List[Dict[str, Any]]:
    conn = _get_connection()
    cursor = conn.cursor()

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
    conn.close()
    return [_row_to_log(row) for row in rows]
