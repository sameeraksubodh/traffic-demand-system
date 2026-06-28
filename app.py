from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import pickle
import os

# Point to root folder for HTML structure assets
app = Flask(__name__, template_folder='.')

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
            return jsonify({"error": "Empty payload received"}), 400

        # 1. ROBUST STRUCTURAL TYPE PARSING & FALLBACKS
        hour = int(req.get("hour", 10))
        lanes = int(req.get("active_lanes", 4))
        
        # Safely extract dynamic floats
        try:
            lag_1 = float(req.get("lag_1", 210.0))
        except (ValueError, TypeError):
            lag_1 = 210.0
            
        try:
            lag_2 = float(req.get("lag_2", 195.0))
        except (ValueError, TypeError):
            lag_2 = 195.0

        # 2. EVALUATE RUSH INDICES
        is_peak = 1 if (7 <= hour <= 9 or 17 <= hour <= 19) else 0

        # 3. CONVERT FRONTEND TEXT SELECTIONS TO NUMERIC MODIFIERS
        road_classification = req.get("road_classification", "Express Highway")
        road_type_map = {"Express Highway": 1, "Arterial Route": 2, "Urban Commuter Route": 3}
        road_type = road_type_map.get(road_classification, 2)
        road_mod = 1.25 if road_classification == "Express Highway" else 1.0

        weather_state = req.get("weather_state", "Clear Skies")
        if weather_state == "Heavy Storms":
            rainfall, visibility, weather_mod = 12.5, 1.5, 1.3
        elif weather_state == "Active Rainfall":
            rainfall, visibility, weather_mod = 4.2, 5.0, 1.1
        else:
            rainfall, visibility, weather_mod = 0.0, 10.0, 1.0

        # 4. EXECUTE PROCESSING MACHINE LEARNING PIPELINE VS FALLBACK
        if stack and "lgb" in stack:
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
            base_demand = ((lag_1 + lag_2) / 2.0) * road_mod * weather_mod
            if is_peak:
                base_demand += 140.0
            prediction = max(int(base_demand), 15)

        # 5. GENERATE STRUCTURAL FORECAST MATRIX OUTPUTS
        theoretical_capacity = lanes * 1500  
        surge_buffer = max(theoretical_capacity - prediction, 0)
        pct = (prediction / theoretical_capacity) * 100.0

        if pct > 75.0:
            routing = "🚨 High network saturation detected on this corridor. Divert oncoming vehicle streams to secondary peripheral bypass channels immediately."
        elif pct > 45.0:
            routing = "⚠️ Moderate volume build-up observed. Recommend micro-adjusting inbound ramp-metering countdown timers to mitigate localized queue delays."
        else:
            routing = "✅ Traffic stream is moving smoothly within standard operational network bounds. No structural route interventions required."

        return jsonify({
            "predicted_vehicle_count": prediction,
            "pcu_equivalency": round(prediction * 1.15, 1),
            "surge_ceiling_buffer": surge_buffer,
            "total_capacity": theoretical_capacity,
            "smart_routing_recommendation": routing
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
