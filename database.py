import sqlite3
import random
from datetime import datetime, timedelta

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
