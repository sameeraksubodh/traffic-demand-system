from flask import Flask, request, jsonify, render_template
import numpy as np

# Pointing template asset references directly to root project space
app = Flask(__name__, template_folder='.')

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        req = request.get_json()
        if not req:
            return jsonify({"error": "Empty payload string parameter matrix received"}), 400

        # 1. STRUCTURAL DATA TYPECAST EXTRACTION ENGINE
        hour = int(float(req.get("hour", 10)))
        lanes = int(float(req.get("active_lanes", 4)))
        lag_1 = float(req.get("lag_1", 210.0))
        lag_2 = float(req.get("lag_2", 195.0))

        # Enforce structural evaluation bounds protection constraints
        hour = min(max(hour, 0), 23)
        lanes = min(max(lanes, 1), 8)
        lag_1 = max(lag_1, 10.0)
        lag_2 = max(lag_2, 10.0)

        # 2. CALCULATE COMMUTER PEAK TRAFFIC MODIFIERS
        is_peak = 1 if ((7 <= hour <= 9) or (17 <= hour <= 19)) else 0

        # 3. INTERPRET VIEW FIELD SELECTION COEFFICIENTS
        road_classification = str(req.get("road_classification", "Express Highway")).strip()
        road_mod = 1.35 if "Express" in road_classification else 1.10 if "Arterial" in road_classification else 0.85

        weather_state = str(req.get("weather_state", "Clear Skies")).strip()
        if "Storm" in weather_state:
            weather_mod = 1.30
        elif "Rain" in weather_state:
            weather_mod = 1.15
        else:
            weather_mod = 1.00

        # 4. HIGH-FIDELITY COMBINATORIAL PREDICTION ESTIMATION MACHINE LOGIC
        # Blending weighted rolling averages with peak demand multipliers and environmental weather constraints
        historical_baseline = (lag_1 * 0.55) + (lag_2 * 0.45)
        peak_volume_impact = 195.0 if is_peak else 0.0
        
        calculated_demand = (historical_baseline * road_mod * weather_mod) + peak_volume_impact
        
        # Safe structural bounds clipping to ensure it is logically bounded
        theoretical_capacity = lanes * 1500  
        prediction = min(max(int(calculated_demand), 25), theoretical_capacity)

        # 5. INTEGRATED METRIC MATRIX COEFFICIENTS FORMULAS
        pcu_val = round(prediction * 1.18, 1) # Standard Passenger Car Unit conversion scalar factor
        surge_buffer = max(theoretical_capacity - prediction, 0)
        pct = round((prediction / theoretical_capacity) * 100, 1)

        # Contextual Smart Routing Suggestions Engine
        if pct > 75.0:
            routing = "🚨 High network saturation detected on this path. Divert oncoming traffic flow to secondary peripheral bypass channels immediately."
        elif pct > 45.0:
            routing = "⚠️ Moderate volume building. Recommend micro-adjusting ramp-metering countdown timers on inbound lanes to mitigate queue build-up."
        else:
            routing = "✅ Traffic stream is moving smoothly within normal operational network bounds. No route interventions required."

        return jsonify({
            "predicted_vehicle_count": prediction,
            "pcu_equivalency": pcu_val,
            "surge_ceiling_buffer": surge_buffer,
            "total_capacity": theoretical_capacity,
            "smart_routing_recommendation": routing
        }), 200

    except Exception as e:
        return jsonify({"error": f"Internal structural evaluation fault exception: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
