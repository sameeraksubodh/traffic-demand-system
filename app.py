from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import numpy as np
import pickle
import os
import datetime
from database import init_db, log_prediction, get_all_segments, get_analytics_summary

app = Flask(__name__)
init_db()

# Load Model Framework Stack Safely
ARTIFACT_PATH = "innovexa_traffic_stack.pkl"
if os.path.exists(ARTIFACT_PATH):
    with open(ARTIFACT_PATH, "rb") as f:
        stack = pickle.load(f)
else:
    stack = None

def compute_advanced_features(input_dict):
    """Dynamically reconstruct blueprint advanced feature pipeline transforms matrix"""
    # Parse core datetime values
    dt = datetime.datetime.fromisoformat(input_dict["timestamp"].replace("Z", ""))
    hour = dt.hour
    day_of_week = dt.weekday()
    month = dt.month
    weekend = 1 if day_of_week >= 5 else 0
    
    peak_hour = 1 if hour in [8, 9, 17, 18, 19] else 0
    rush_score = peak_hour * 3 + (1 - weekend) * 2
    
    # Weather calculations
    rainfall = float(input_dict.get("rainfall", 0.0))
    visibility = float(input_dict.get("visibility", 10.0))
    wind_speed = float(input_dict.get("wind_speed", 10.0))
    weather_impact = (rainfall * 5) + (10 - visibility) + (wind_speed * 0.2)
    
    # Construct base dictionary parameters
    feats = {
        "Road_Segment_ID": int(input_dict["segment_id"]),
        "Road_Type": int(input_dict.get("road_type", 2)),
        "Number_of_Lanes": int(input_dict.get("number_of_lanes", 3)),
        "Speed_Limit": int(input_dict.get("speed_limit", 60)),
        "Temperature": float(input_dict.get("temperature", 25.0)),
        "Humidity": float(input_dict.get("humidity", 60.0)),
        "Rainfall": rainfall,
        "Visibility": visibility,
        "Wind_Speed": wind_speed,
        "Nearby_POI_Density": float(input_dict.get("poi_density", 45.0)),
        "Event_Holiday": int(input_dict.get("event_holiday", 0)),
        "Peak_Hour_Indicator": peak_hour,
        "Rush_Hour_Score": rush_score,
        "Hour_Sin": np.sin(2 * np.pi * hour / 24.0),
        "Hour_Cos": np.cos(2 * np.pi * hour / 24.0),
        "Day_Sin": np.sin(2 * np.pi * day_of_week / 7.0),
        "Day_Cos": np.cos(2 * np.pi * day_of_week / 7.0),
        "Month_Sin": np.sin(2 * np.pi * month / 12.0),
        "Month_Cos": np.cos(2 * np.pi * month / 12.0),
        "Weather_Impact_Score": weather_impact,
        "Weather_x_Hour": weather_impact * hour,
        "Lag_1": float(input_dict.get("lag_1", 150.0)),
        "Lag_2": float(input_dict.get("lag_2", 145.0)),
        "Lag_3": float(input_dict.get("lag_3", 140.0)),
        "Lag_4": float(input_dict.get("lag_4", 140.0)),
        "Lag_6": float(input_dict.get("lag_6", 135.0)),
        "Lag_12": float(input_dict.get("lag_12", 130.0)),
        "Rolling_Mean_3": float(input_dict.get("rolling_mean_3", 148.0)),
        "Rolling_Std_3": float(input_dict.get("rolling_std_3", 5.0)),
        "Rolling_Mean_6": float(input_dict.get("rolling_mean_6", 144.0)),
        "Rolling_Mean_12": float(input_dict.get("rolling_mean_12", 140.0)),
        "Rolling_Mean_24": float(input_dict.get("rolling_mean_24", 138.0)),
        "EMA": float(input_dict.get("ema", 146.0))
    }
    return pd.DataFrame([feats])[stack["feature_cols"]]

# --- FRONTEND INTERACTIVE UI TEMPLATE ---
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8"><title>Innovexa Catalyst - Real-Time Dashboard</title>
    <link href="https://jsdelivr.net" rel="stylesheet">
</head>
<body class="bg-light">
    <nav class="navbar navbar-dark bg-dark shadow-sm"><div class="container"><span class="navbar-brand mb-0 h1">🚦 INNOVEXA CATALYST TRAFFIC INTELLIGENCE MANAGEMENT CENTER</span></div></nav>
    <div class="container mt-4">
        <div class="row g-4">
            <div class="col-lg-5">
                <div class="card shadow-sm border-0 mb-4">
                    <div class="card-header bg-secondary text-white fw-bold">⚙️ Predictive Analytics Workspace Control</div>
                    <div class="card-body">
                        <form id="trafficForm" class="row g-3">
                            <div class="col-12"><label class="form-label">Road Segment Configuration Profile</label><select class="form-select" name="segment_id" id="segSelect" required><option value="101">Segment 101 - Core Highway Arterial</option><option value="102">Segment 102 - Urban Commuter Route</option><option value="103">Segment 103 - Commercial Sector Center</option><option value="104">Segment 104 - Regional Logistics Gateway</option></select></div>
                            <div class="col-12"><label class="form-label">Target Evaluation Timestamp</label><input type="datetime-local" class="form-control" name="timestamp" id="timeInput" required></div>
                            <div class="col-md-6"><label class="form-label">Temperature (°C)</label><input type="number" class="form-control" name="temperature" value="27" required></div>
                            <div class="col-md-6"><label class="form-label">Rainfall Velocity (mm)</label><input type="text" class="form-control" name="rainfall" value="0.0" required></div>
                            <div class="col-md-6"><label class="form-label">Visibility Index (km)</label><input type="number" class="form-control" name="visibility" value="10" required></div>
                            <div class="col-md-6"><label class="form-label">Event Active Status</label><select class="form-select" name="event_holiday"><option value="0">Standard Schedule Day</option><option value="1">Active Event/Holiday Area</option></select></div>
                            <button type="submit" class="btn btn-primary w-100 mt-3 fw-bold">Run Stacking Ensemble Evaluation Pipeline</button>
                        </form>
                    </div>
                </div>
            </div>
            <div class="col-lg-7">
                <div class="card shadow-sm border-0 mb-4 p-4 d-none alert" id="alertBox">
                    <h4 class="alert-heading m-0">Evaluation Output Stream:</h4><hr>
                    <div class="d-flex justify-content-between align-items-center"><span class="fs-4">Predicted Volume Stream Target:</span><strong class="display-6" id="volOut">0</strong></div>
                    <div class="d-flex justify-content-between align-items-center mt-2"><span class="fs-5">Calculated Congestion Alert Status:</span><span class="badge fs-6" id="badgeOut">Normal</span></div>
                </div>
                <div class="card shadow-sm border-0 p-4 bg-white">
                    <h5 class="fw-bold text-dark mb-3">📈 Live System Telemetry Engine Logs Summary</h5>
                    <div class="row g-2 text-center" id="telemetryData">
                        <div class="col-6 bg-light p-3 border rounded"><h6>Total Calculations</h6><h4 id="telTotal">0</h4></div>
                        <div class="col-6 bg-light p-3 border rounded"><h6>Mean Vol Run</h6><h4 id="telAvg">0.0</h4></div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    <script>
        document.getElementById('timeInput').value = new Date().toISOString().slice(0,16);
        async function fetchTelemetry() {
            const r = await fetch('/api/analytics'); const j = await r.json();
            document.getElementById('telTotal').innerText = j.total_predictions_evaluated;
            document.getElementById('telAvg').innerText = j.average_predicted_volume;
        }
        document.getElementById('trafficForm').onsubmit = async(e) => {
            e.preventDefault();
            const d = Object.fromEntries(new FormData(e.target).entries());
            const r = await fetch('/api/predict', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(d) });
            const j = await r.json();
            const box = document.getElementById('alertBox'); box.classList.remove('d-none','alert-success','alert-warning','alert-danger');
            document.getElementById('volOut').innerText = j.predicted_vehicle_count + " Vehicles";
            const b = document.getElementById('badgeOut'); b.innerText = j.prediction_alert_level;
            if(j.predicted_vehicle_count > 450) { box.classList.add('alert-danger'); b.className='badge bg-danger'; }
            else if(j.predicted_vehicle_count > 220) { box.classList.add('alert-warning'); b.className='badge bg-warning text-dark'; }
            else { box.classList.add('alert-success'); b.className='badge bg-success'; }
            fetchTelemetry();
        };
        fetchTelemetry();
    </script>
</body>
</html>
"""

@app.route('/', methods=['GET'])
def index():
    return render_template_string(DASHBOARD_HTML)

# --- SATISFYING ALL 5 REQUIRED REST API ENDPOINTS (Section 7 Blueprint Specs) ---

@app.route('/api/health', methods=['GET'])
def health():
    """GET /api/health"""
    return jsonify({
        "status": "healthy",
        "engine_timestamp": datetime.datetime.now().isoformat(),
        "ensemble_weights": {"LightGBM": 0.55, "XGBoost": 0.45},
        "model_loaded_and_active": stack is not None
    }), 200

@app.route('/api/segments', methods=['GET'])
def segments():
    """GET /api/segments"""
    return jsonify(get_all_segments()), 200

@app.route('/api/analytics', methods=['GET'])
def analytics():
    """GET /api/analytics"""
    return jsonify(get_analytics_summary()), 200

@app.route('/api/predict', methods=['POST'])
def predict():
    """POST /api/predict"""
    if not stack:
        return jsonify({"error": "Model payload uninitialized."}), 500
    try:
        req_data = request.get_json()
        df_features = compute_advanced_features(req_data)
        
        # Stacking execution calculation parameters pipeline
        p_lgb = stack["lgb"].predict(df_features)[0]
