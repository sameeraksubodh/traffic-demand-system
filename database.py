import sqlite3
import datetime

DB_NAME = "traffic_analytics.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS system_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            segment_id INTEGER,
            predicted_demand INTEGER,
            congestion_index REAL,
            alert_status TEXT
        )
    ''')
    # Pre-populate distinct infrastructure segment configurations metadata references
    cursor.execute('DROP TABLE IF EXISTS infrastructure_segments')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS infrastructure_segments (
            segment_id INTEGER PRIMARY KEY,
            road_type INTEGER,
            lanes INTEGER,
            speed_limit INTEGER,
            poi_density REAL
        )
    ''')
    segments_data = [
        (101, 1, 4, 100, 45.2),
        (102, 2, 3, 80, 72.1),
        (103, 2, 2, 50, 85.0),
        (104, 3, 2, 50, 15.4)
    ]
    cursor.executemany('INSERT OR IGNORE INTO infrastructure_segments VALUES (?,?,?,?,?)', segments_data)
    conn.commit()
    conn.close()

def log_prediction(segment_id, val, congestion, alert):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO system_logs (timestamp, segment_id, predicted_demand, congestion_index, alert_status)
        VALUES (?, ?, ?, ?, ?)
    ''', (datetime.datetime.now().isoformat(), segment_id, val, congestion, alert))
    conn.commit()
    conn.close()

def get_all_segments():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM infrastructure_segments")
    rows = cursor.fetchall()
    conn.close()
    return [{"segment_id": r[0], "road_type": r[1], "lanes": r[2], "speed_limit": r[3], "poi_density": r[4]} for r in rows]

def get_analytics_summary():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*), AVG(predicted_demand), MAX(predicted_demand) FROM system_logs")
    summary = cursor.fetchone()
    cursor.execute("SELECT segment_id, COUNT(*) FROM system_logs GROUP BY segment_id ORDER BY COUNT(*) DESC LIMIT 1")
    top_seg = cursor.fetchone()
    conn.close()
    return {
        "total_predictions_evaluated": summary[0] if summary[0] else 0,
        "average_predicted_volume": round(summary[1], 2) if summary[1] else 0.0,
        "max_peak_volume_detected": summary[2] if summary[2] else 0,
        "most_active_segment_id": top_seg[0] if top_seg else "N/A"
    }
