from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import pickle
import os

# Point to root folder for HTML template assets
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

        # 1. ENFORCE STRICT SERVER-SIDE TYPECASTING (Fixes the Silent Mismatch Mismatch)
        hour = int(float(req.get("hour", 10)))
        lanes = int(float(req.get("active_lanes", 4)))
        lag_1 = float(req.get("lag_1", 210.0))
        lag_2 = float(req.get("lag_2", 195.0))

        # Enforce realistic bounds to protect model arrays from math scaling failure
        hour = min(max(hour, 0), 23)
        lanes = min(max(lanes, 1), 8)
        lag_1 = max(lag_1, 0.0)
        lag_2 = max(lag_2, 0.0)

        # 2. EVALUATE TRAFFIC PEAK HOUR INDICES
        is_peak = 1 if (7 <= hour <= 9 or 17 <= hour <= 19) else 0

        # 3. CONVERT FRONTEND OPTION CHOICES TO SYSTEM COEFFICIENTS
        road_classification = str(req.get("road_classification", "Express Highway")).strip()
        road_type_map = {
            "Express Highway": 1, 
            "Arterial Route": 2, 
            "Urban Commuter Route": 3,
            "Urban Commuter": 3
        }
        road_type = road_type_map.get(road_classification, 2)
        road_mod = 1.35 if "Express" in road_classification else 1.0

        weather_state = str(req.get("weather_state", "Clear Skies")).strip()
        if "Storm" in weather_state:
            rainfall, visibility, weather_mod = 12.5, 1.5, 1.25
        elif "Rain" in weather_state:
            rainfall, visibility, weather_mod = 4.2, 5.0, 1.12
        else:
            rainfall, visibility, weather_mod = 0.0, 10.0, 1.0

        # 4. INFERENCE EXECUTION BLOCK (Ensembles Stacking Predictions vs Robust Math Equations)
        if stack and "lgb" in stack and hasattr(stack, "feature_cols"):
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
            # High-Fidelity Math Fallback calculation engine logic
            historical_base = (lag_1 * 0.6) + (lag_2 * 0.4)
            peak_multiplier = 185 if is_peak else 0
            raw_calculated_volume = (historical_base * road_mod * weather_mod) + peak_multiplier
            prediction = max(int(raw_calculated_volume), 25)

        # 5. INTEGRATE ANALYTICS MATRIX SYSTEM CALCULATIONS
        theoretical_capacity = lanes * 1500  
        prediction = min(prediction, theoretical_capacity) # Cap demand by physical lane capacity max limits
        
        pcu_val = round(prediction * 1.18, 1) # Standard Passenger Car Unit passenger vehicle conversion
        surge_buffer = max(theoretical_capacity - prediction, 0)
        pct = round((prediction / theoretical_capacity) * 100, 1)

        if pct > 75.0:
            routing = "🚨 High saturation detected on this path. Divert oncoming traffic flow to secondary peripheral bypass roads immediately."
        elif pct > 45.0:
            routing = "⚠️ Moderate volume building. Recommend micro-adjusting ramp-metering timers on inbound lanes."
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
        # Avoid zero blank silences; send exact execution error context to web console tools
        return jsonify({"error": f"Internal execution crash fault: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
