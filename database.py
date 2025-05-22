import sqlite3
import datetime
from dotenv import load_dotenv
import os

load_dotenv()


db_path = os.getenv('SQLITE_DB_PATH', 'reports.db')

def setup_database():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            object TEXT,
            confidence REAL,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()

def log_detection(object_name, confidence):
    conn = sqlite3.connect(db_path)
    """
    Logs a detection event to the database.

    Parameters:
    object_name (str): The name of the detected object.
    confidence (float): The confidence score of the detection.

    This function inserts a new record into the 'detections' table
    with the specified object name and confidence score.
    """

    cursor = conn.cursor()
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("INSERT INTO detections (object, confidence, timestamp) VALUES (?, ?, ?)", 
                   (object_name, confidence, timestamp))
    
    conn.commit()
    conn.close()
    print(f"Logged: {object_name} ({confidence:.2f}) at {timestamp}")

setup_database()