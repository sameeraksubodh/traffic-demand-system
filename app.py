from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import pickle
import os

app = Flask(__name__)

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
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        req = request.get_json()
        if not req:
            return jsonify({"error": "Empty payload"}), 400

        # Enforce server-side explicit casting to fix formatting errors
        hour = int(float(req.get("hour", 10)))
        lanes = int(float(req.get("active_lanes", 4)))
        lag_1 = float(req.get("lag_1", 210.0))
        lag_2 = float(req.get("lag_2", 195.0))

        is_peak = 1 if (7 <= hour <= 9 or 17 <= hour <= 19) else 0
        road_classification = str(req.get("road_classification", "Express Highway"))
        weather_state = str(req.get("weather_state", "Clear Skies"))

        road_mod = 1.35 if "Express" in road_classification else 1.0
        weather_mod = 1.25 if "Storm" in weather_state else 1.1 if "Rain" in weather_state else 1.0

        if stack and "lgb" in stack:
            road_type = 1 if "Express" in road_classification else 2 if "Arterial" in road_classification else 3
            rainfall = 12.5 if "Storm" in weather_state else 4.2 if "Rain" in weather_state else 0.0
            visibility = 1.5 if "Storm" in weather_state else 5.0 if "Rain" in weather_state else 10.0
            weather_impact = (rainfall * 5.0) + (10.0 - visibility) + 2.0
            
            feats = {
                "Road_Segment_ID": 101, "Road_Type": road_type, "Number_of_Lanes": lanes, "Speed_Limit": 80,
                "Temperature": 26.0, "Humidity": 75.0, "Rainfall": rainfall, "Visibility": visibility, "Wind_Speed": 12.0,
                "Nearby_POI_Density": 45.0, "Event_Holiday": 0, "Peak_Hour_Indicator": is_peak, "Rush_Hour_Score": (is_peak * 3) + 2,
                "Hour_Sin": np.sin(2 * np.pi * hour / 24.0), "Hour_Cos": np.cos(2 * np.pi * hour / 24.0),
                "Day_Sin": 0.0, "Day_Cos": 1.0, "Month_Sin": 0.0, "Month_Cos": 1.0,
                "Weather_Impact_Score": weather_impact, "Weather_x_Hour": weather_impact * hour,
                "Lag_1": lag_1, "Lag_2": lag_2, "Lag_3": lag_2, "Lag_4": lag_2, "Lag_6": lag_2, "Lag_12": lag_2,
                "Rolling_Mean_3": (lag_1 + lag_2) / 2.0, "Rolling_Std_3": 5.0, "Rolling_Mean_6": lag_2, "Rolling_Mean_12": lag_2, "Rolling_Mean_24": lag_2, "EMA": lag_1
            }
            df = pd.DataFrame([feats])[stack["feature_cols"]]
            prediction = max(int(stack["meta"].predict(np.column_stack((stack["lgb"].predict(df), stack["xgb"].predict(df))))), 15)
        else:
            base_calc = ((lag_1 + lag_2) / 2.0) * road_mod * weather_mod
            if is_peak:
                base_calc += 140.0
            prediction = max(int(base_calc), 15)

        theoretical_capacity = lanes * 1500  
        prediction = min(prediction, theoretical_capacity)
        
        pcu_val = round(prediction * 1.15, 1)
        surge_buffer = max(theoretical_capacity - prediction, 0)
        pct = (prediction / theoretical_capacity) * 100.0
        
        if pct > 75.0:
            routing = "🚨 High saturation detected on this path. Divert oncoming traffic flow to secondary peripheral bypass channels immediately."
        elif pct > 45.0:
            routing = "⚠️ Moderate volume building. Recommend micro-adjusting ramp-metering countdown timers on inbound lanes."
        else:
            routing = "✅ Traffic flowing smoothly within normal operational parameters. No route interventions required."

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
