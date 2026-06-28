from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import numpy as np
import pickle
import os
from templates import DASHBOARD_HTML

app = Flask(__name__)

# --- SAFE PRODUCTION ARTIFACT INITIALIZER ---
def load_ml_stack():
    artifact_path = "innovexa_traffic_stack.pkl"
    if not os.path.exists(artifact_path):
        return None
    try:
        import lightgbm
        import xgboost
        with open(artifact_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        return None

stack = load_ml_stack()

@app.route('/', methods=['GET'])
def index():
    return render_template_string(DASHBOARD_HTML)

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        req = request.get_json()
        hour = int(req["hour"])
        lanes = int(req["active_lanes"])
        lag_1 = float(req["lag_1"])
        lag_2 = float(req["lag_2"])
        
        if hour >= 7 and hour <= 9:
            peak_hour = 1
        elif hour >= 17 and hour <= 19:
            peak_hour = 1
        else:
            peak_hour = 0
            
        if stack and "lgb" in stack:
            road_type_map = {"Express Highway": 1, "Arterial Route": 2, "Urban Commuter": 3}
            road_type = road_type_map.get(req["road_classification"], 2)
            weather_state = req.get("weather_state", "Clear Skies")
            rainfall = 0.0 if weather_state == "Clear Skies" else 4.2 if weather_state == "Active Rainfall" else 12.5
            visibility = 10.0 if weather_state == "Clear Skies" else 5.0 if weather_state == "Active Rainfall" else 1.5
            weather_impact = (rainfall * 5.0) + (10.0 - visibility) + 2.0
            rush_score = (peak_hour * 3) + 2
            
            feats = {
                "Road_Segment_ID": 101, "Road_Type": road_type, "Number_of_Lanes": lanes, "Speed_Limit": 80,
                "Temperature": 26.0, "Humidity": 75.0, "Rainfall": rainfall, "Visibility": visibility, "Wind_Speed": 12.0,
                "Nearby_POI_Density": 45.0, "Event_Holiday": 0, "Peak_Hour_Indicator": peak_hour, "Rush_Hour_Score": rush_score,
                "Hour_Sin": np.sin(2 * np.pi * hour / 24.0), "Hour_Cos": np.cos(2 * np.pi * hour / 24.0),
                "Day_Sin": 0.0, "Day_Cos": 1.0, "Month_Sin": 0.0, "Month_Cos": 1.0,
                "Weather_Impact_Score": weather_impact, "Weather_x_Hour": weather_impact * hour,
                "Lag_1": lag_1, "Lag_2": lag_2, "Lag_3": lag_2, "Lag_4": lag_2, "Lag_6": lag_2, "Lag_12": lag_2,
                "Rolling_Mean_3": (lag_1 + lag_2) / 2.0, "Rolling_Std_3": 5.0, "Rolling_Mean_6": lag_2,
                "Rolling_Mean_12": lag_2, "Rolling_Mean_24": lag_2, "EMA": lag_1
            }
            df = pd.DataFrame([feats])[stack["feature_cols"]]
            p_lgb = stack["lgb"].predict(df)
            p_xgb = stack["xgb"].predict(df)
            meta_in = np.column_stack((p_lgb, p_xgb))
            prediction = max(int(stack["meta"].predict(meta_in)), 15)
        else:
            weather_mod = 1.3 if req["weather_state"] == "Heavy Storms" else 1.1 if req["weather_state"] == "Active Rainfall" else 1.0
            road_mod = 1.25 if req["road_classification"] == "Express Highway" else 1.0
            base_demand = ((lag_1 + lag_2) / 2.0) * road_mod * weather_mod
            if peak_hour:
                base_demand += 140.0
            prediction = max(int(base_demand), 15)
        
        pcu_val = round(prediction * 1.15, 1)
        theoretical_capacity = lanes * 1500  
        surge_buffer = max(theoretical_capacity - prediction, 0)
        pct = (prediction / theoretical_capacity) * 100.0
        
        if pct > 75.0:
            routing = "High saturation detected on this path. Divert oncoming traffic flow to secondary peripheral bypass roads immediately."
        elif pct > 45.0:
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
