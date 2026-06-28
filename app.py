from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import numpy as np
import pickle
import os

app = Flask(__name__)

# Load Model Framework Stack Safely
ARTIFACT_PATH = "innovexa_traffic_stack.pkl"
if os.path.exists(ARTIFACT_PATH):
    with open(ARTIFACT_PATH, "rb") as f:
        stack = pickle.load(f)
else:
    stack = None

# --- HTML/JS INTERACTIVE UI ---
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Intelligent Live Traffic Management Dashboard</title>
    <link href="https://jsdelivr.net" rel="stylesheet">
    <style>
        .card { border-radius: 12px; border: 1px solid #eef2f5; box-shadow: 0 4px 12px rgba(0,0,0,0.03); }
        .btn-primary { background-color: #1e56df; border: none; padding: 12px; font-weight: bold; }
        .btn-primary:hover { background-color: #1542b3; }
        .form-label { font-weight: 500; color: #495057; }
    </style>
</head>
<body class="bg-light py-5">
    <div class="container" style="max-width: 900px;">
        <div class="card p-4 mb-4 bg-white">
            <h4 class="mb-4 d-flex align-items-center">
                <span class="me-2">🚦</span> <strong>Intelligent Live Traffic Management Dashboard</strong>
            </h4>
            
            <form id="trafficForm" class="row g-3">
                <div class="col-md-6">
                    <label class="form-label">Hour of Day (0-23)</label>
                    <input type="number" class="form-control" name="hour" min="0" max="23" value="10" required>
                </div>
                <div class="col-md-6">
                    <label class="form-label">Road Classification</label>
                    <select class="form-select" name="road_classification">
                        <option value="Express Highway">Express Highway</option>
                        <option value="Arterial Route">Arterial Route</option>
                        <option value="Urban Commuter">Urban Commuter Route</option>
                    </select>
                </div>
                <div class="col-md-6">
                    <label class="form-label">Active Lanes</label>
                    <input type="number" class="form-control" name="active_lanes" min="1" max="10" value="4" required>
                </div>
                <div class="col-md-6">
                    <label class="form-label">Weather State</label>
                    <select class="form-select" name="weather_state">
                        <option value="Clear Skies">Clear Skies</option>
                        <option value="Active Rainfall">Active Rainfall</option>
                        <option value="Heavy Storms">Heavy Storms</option>
                    </select>
                </div>
                <div class="col-md-6">
                    <label class="form-label">Lag 1 Traffic Volume</label>
                    <input type="number" class="form-control" name="lag_1" value="210" required>
                </div>
                <div class="col-md-6">
                    <label class="form-label">Lag 2 Traffic Volume</label>
                    <input type="number" class="form-control" name="lag_2" value="195" required>
                </div>
                <div class="col-12 mt-4">
                    <button type="submit" class="btn btn-primary w-100 text-uppercase tracking-wide">Execute Stacking Analytics</button>
                </div>
            </form>
        </div>

        <!-- Output Forecast Area -->
        <div class="card p-4 mb-4 bg-white">
            <h6 class="text-muted text-uppercase fw-bold small">Ensemble Forecasted Volume:</h6>
            <div class="display-5 fw-bold text-dark mt-2" id="volOut">0</div>
        </div>

        <!-- Metric Details Grid Matrix -->
        <div class="row g-4">
            <div class="col-md-6">
                <div class="card p-4 bg-white h-100">
                    <h6 class="fw-bold d-flex align-items-center mb-3">📊 Capacity Analytics Matrix</h6>
                    <p class="mb-1 text-muted">PCU Equivalency: <strong class="text-dark" id="pcuOut">0</strong></p>
                    <p class="mb-3 text-muted">Surge Ceiling Buffer: <strong class="text-dark" id="bufferOut">0</strong></p>
                    <div class="progress" style="height: 8px;">
                        <div class="progress-bar bg-primary" id="progressFill" style="width: 0%"></div>
                    </div>
                    <small class="text-muted d-block mt-2 text-end" id="progressPct">0%</small>
                </div>
            </div>
            <div class="col-md-6">
                <div class="card p-4 bg-white h-100">
                    <h6 class="fw-bold d-flex align-items-center mb-3">🧠 Smart Routing Recommendation</h6>
                    <p class="text-secondary" id="routingOut">Waiting for structural data stream execution...</p>
                </div>
            </div>
        </div>
    </div>

    <script>
        document.getElementById('trafficForm').onsubmit = async (e) => {
            e.preventDefault();
            const payload = Object.fromEntries(new FormData(e.target).entries());
            
            const res = await fetch('/api/predict', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(payload)
            });
            const data = await res.json();
            
            document.getElementById('volOut').innerText = data.predicted_vehicle_count + " vehicles";
            document.getElementById('pcuOut').innerText = data.pcu_equivalency;
            document.getElementById('bufferOut').innerText = data.surge_ceiling_buffer;
            
            const pct = Math.min(Math.round((data.predicted_vehicle_count / data.total_capacity) * 100), 100);
            document.getElementById('progressFill').style.width = pct + "%";
            document.getElementById('progressPct').innerText = pct + "%";
            
            document.getElementById('routingOut').innerText = data.smart_routing_recommendation;
        };
    </script>
</body>
</html>
"""

@app.route('/', methods=['GET'])
def index():
    return render_template_string(DASHBOARD_HTML)

@app.route('/api/predict', methods=['POST'])
def predict():
    if not stack:
        return jsonify({"error": "Model payload uninitialized."}), 500
    try:
        req = request.get_json()
        
        # Map input choices safely to numeric variables
        road_type_map = {"Express Highway": 1, "Arterial Route": 2, "Urban Commuter": 3}
        road_type = road_type_map.get(req["road_classification"], 2)
        
        weather_state = req.get("weather_state", "Clear Skies")
        rainfall = 0.0 if weather_state == "Clear Skies" else 4.2 if weather_state == "Active Rainfall" else 12.5
        visibility = 10.0 if weather_state == "Clear Skies" else 5.0 if weather_state == "Active Rainfall" else 1.5
        
        hour = int(req["hour"])
        lanes = int(req["active_lanes"])
        lag_1 = float(req["lag_1"])
        lag_2 = float(req["lag_2"])
        
        weather_impact = (rainfall * 5) + (10 - visibility) + 2.0
        
        # FIXED: Removed broken blank bracket array assignment logic entirely
        peak_hour = 1 if (7 <= hour <= 9 or 17 <= hour <= 19) else 0
        rush_score = peak_hour * 3 + 2
        
        feats = {
            "Road_Segment_ID": 101, "Road_Type": road_type, "Number_of_Lanes": lanes, "Speed_Limit": 80,
            "Temperature": 26.0, "Humidity": 75.0, "Rainfall": rainfall, "Visibility": visibility, "Wind_Speed": 12.0,
            "Nearby_POI_Density": 45.0, "Event_Holiday": 0, "Peak_Hour_Indicator": peak_hour, "Rush_Hour_Score": rush_score,
            "Hour_Sin": np.sin(2 * np.pi * hour / 24.0), "Hour_Cos": np.cos(2 * np.pi * hour / 24.0),
            "Day_Sin": 0.0, "Day_Cos": 1.0, "Month_Sin": 0.0, "Month_Cos": 1.0,
            "Weather_Impact_Score": weather_impact, "Weather_x_Hour": weather_impact * hour,
            "Lag_1": lag_1, "Lag_2": lag_2, "Lag_3": lag_2, "Lag_4": lag_2, "Lag_6": lag_2, "Lag_12": lag_2,
            "Rolling_Mean_3": (lag_1 + lag_2) / 2, "Rolling_Std_3": 5.0, "Rolling_Mean_6": lag_2,
            "Rolling_Mean_12": lag_2, "Rolling_Mean_24": lag_2, "EMA": lag_1
        }
        
        df = pd.DataFrame([feats])[stack["feature_cols"]]
        
        # Model Prediction
        p_lgb = stack["lgb"].predict(df)
        p_xgb = stack["xgb"].predict(df)
        meta_in = np.column_stack((p_lgb, p_xgb))
        prediction = max(int(stack["meta"].predict(meta_in)), 15)
        
        pcu_val = round(prediction * 1.15, 1)
        theoretical_capacity = lanes * 1500  
        surge_buffer = max(theoretical_capacity - prediction, 0)
        pct = (prediction / theoretical_capacity) * 100
        
        if pct > 75:
            routing = "High saturation detected on this path. Divert oncoming traffic flow to secondary peripheral bypass roads immediately."
        elif pct > 45:
            routing = "Moderate volume building. Recommend micro-adjusting ramp-metering timers on inbound lanes."
        else:
            routing = "Traffic flowing smoothly within normal operational parameters. No route interventions required."

        return jsonify({
            "predicted_vehicle_count": prediction,
            "pcu_equivalency": pcu_val,
            "surge_ceiling_buffer": surge_buffer,
            "total_capacity": theoretical_capacity,
            "smart_routing_recommendation": routing
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
