from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import numpy as np
import pickle, os
app = Flask(__name__)

def load_ml_stack():
    p = "innovexa_traffic_stack.pkl"
    if not os.path.exists(p): return None
    try:
        import lightgbm, xgboost
        with open(p, "rb") as f: return pickle.load(f)
    except: return None
stack = load_ml_stack()

# Streamlined layout with a beautiful premium side-by-side arrangement
UI = """
<!DOCTYPE html><html><head><meta charset="UTF-8"><title>Dashboard</title>
<link href="https://jsdelivr.net" rel="stylesheet">
<style>
:root { --bg: #f8fafc; --surface: #ffffff; --text: #0f172a; --border: #e2e8f0; }
body { font-family: system-ui, sans-serif; background: var(--bg); color: var(--text); }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 16px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); padding: 24px; }
.form-control, .form-select { border: 1px solid var(--border); border-radius: 10px; padding: 10px; }
.btn-primary { background: #2563eb; border: none; border-radius: 10px; padding: 12px; font-weight: 600; text-transform: uppercase; }
.metric-value { font-size: 2.5rem; font-weight: 700; color: #2563eb; line-height: 1; }
.nested-metric { background: #f8fafc; border: 1px solid var(--border); border-radius: 12px; padding: 14px; }
.recommendation-box { border-left: 4px solid #2563eb; background: #eff6ff; padding: 16px; border-radius: 0 12px 12px 0; }
</style></head>
<body class="py-4"><nav class="navbar bg-white border-bottom mb-5"><div class="container fw-bold">🚦 Innovexa Catalyst Control Center</div></nav>
<div class="container"><div class="row g-4">
<div class="col-md-5"><div class="card h-100"><h5 class="fw-bold mb-4">Workspace Parameters</h5><form id="trafficForm" class="row g-3">
<div class="col-12"><label class="form-label small fw-bold text-muted">HOUR OF DAY (0-23)</label><input type="number" class="form-control" name="hour" min="0" max="23" value="10" required></div>
<div class="col-12"><label class="form-label small fw-bold text-muted">ROAD TYPE</label><select class="form-select" name="road_classification"><option value="Express Highway">Express Highway</option><option value="Arterial Route">Arterial Route</option><option value="Urban Commuter Route">Urban Commuter Route</option></select></div>
<div class="col-12"><label class="form-label small fw-bold text-muted">ACTIVE LANES</label><input type="number" class="form-control" name="active_lanes" min="1" max="10" value="4" required></div>
<div class="col-12"><label class="form-label small fw-bold text-muted">WEATHER STATE</label><select class="form-select" name="weather_state"><option value="Clear Skies">Clear Skies</option><option value="Active Rainfall">Active Rainfall</option><option value="Heavy Storms">Heavy Storms</option></select></div>
<div class="col-6"><label class="form-label small fw-bold text-muted">LAG 1 (T-15M)</label><input type="number" class="form-control" name="lag_1" value="210" required></div>
<div class="col-6"><label class="form-label small fw-bold text-muted">LAG 2 (T-30M)</label><input type="number" class="form-control" name="lag_2" value="195" required></div>
<div class="col-12 mt-4"><button type="submit" class="btn btn-primary w-100">Run Analytics</button></div>
</form></div></div>
<div class="col-md-7"><div class="d-flex flex-column h-100 gap-4">
<div class="card d-flex flex-column justify-content-center" style="min-height:140px;"><span class="small fw-bold text-muted text-uppercase">Ensemble Forecasted Volume</span><div class="d-flex align-items-baseline gap-2 mt-2"><div class="metric-value" id="volOut">0</div><span class="text-secondary fw-semibold">vehicles / 15-min</span></div></div>
<div class="card"><h6 class="fw-bold mb-3 text-dark">📊 Capacity Analytics Matrix</h6><div class="row g-3 mb-4">
<div class="col-6"><div class="nested-metric"><small class="text-muted fw-bold text-uppercase" style="font-size:0.75rem;">Equivalent Car Space</small><div class="fw-bold fs-5 mt-1" id="pcuOut">0</div></div></div>
<div class="col-6"><div class="nested-metric"><small class="text-muted fw-bold text-uppercase" style="font-size:0.75rem;">Remaining Road Space</small><div class="fw-bold fs-5 mt-1" id="bufferOut">0</div></div></div></div>
<div class="d-flex justify-content-between align-items-center mb-2"><span class="text-secondary small fw-medium">Road Capacity Used</span><span class="fw-bold small" id="progressPct">0%</span></div><div class="progress" style="height:10px;"><div class="progress-bar bg-primary" id="progressFill" style="width:0%"></div></div></div>
<div class="card"><h6 class="fw-bold mb-3 text-dark">🧠 Smart Routing Recommendation</h6><div class="recommendation-box" id="recBox"><p class="mb-0 text-secondary small font-weight-medium" id="routingOut">Waiting for data submission...</p></div></div>
</div></div></div></div>
<script>
document.getElementById('trafficForm').onsubmit = async (e) => {
    e.preventDefault();
    const res = await fetch('/api/predict', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(Object.fromEntries(new FormData(e.target).entries())) });
    const d = await res.json();
    document.getElementById('volOut').innerText = d.predicted_vehicle_count;
    document.getElementById('pcuOut').innerText = d.pcu_equivalency;
    document.getElementById('bufferOut').innerText = d.surge_ceiling_buffer;
    const pct = Math.min(Math.round((d.predicted_vehicle_count / d.total_capacity) * 100), 100);
    document.getElementById('progressPct').innerText = pct + "%";
    const pb = document.getElementById('progressFill'); pb.style.width = pct + "%";
    const rb = document.getElementById('recBox'); const rt = document.getElementById('routingOut'); rt.innerText = d.smart_routing_recommendation;
    if(pct > 75) { pb.className='progress-bar bg-danger'; rb.style.borderColor='#ef4444'; rb.style.background='#fef2f2'; rt.style.color='#991b1b'; }
    else if(pct > 45) { pb.className='progress-bar bg-warning'; rb.style.borderColor='#f59e0b'; rb.style.background='#fffbeb'; rt.style.color='#92400e'; }
    else { pb.className='progress-bar bg-primary'; rb.style.borderColor='#2563eb'; rb.style.background='#eff6ff'; rt.style.color='#1e40af'; }
};
</script></body></html>
"""

@app.route('/', methods=['GET'])
def index(): return render_template_string(UI)

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        req = request.get_json()
        hour, lanes, lag_1, lag_2 = int(float(req["hour"])), int(float(req["active_lanes"])), float(req["lag_1"]), float(req["lag_2"])
        is_peak = 1 if (7 <= hour <= 9 or 17 <= hour <= 19) else 0
        road, weather = str(req.get("road_classification")), str(req.get("weather_state"))
        rmod = 1.35 if "Express" in road else 1.0
        wmod = 1.25 if "Storm" in weather else 1.1 if "Rain" in weather else 1.0
        if stack and "lgb" in stack:
            rtype = 1 if "Express" in road else 2 if "Arterial" in road else 3
            rf = 12.5 if "Storm" in weather else 4.2 if "Rain" in weather else 0.0
            vs = 1.5 if "Storm" in weather else 5.0 if "Rain" in weather else 10.0
            w_imp = (rf * 5.0) + (10.0 - vs) + 2.0
            feats = {
                "Road_Segment_ID": 101, "Road_Type": rtype, "Number_of_Lanes": lanes, "Speed_Limit": 80,
                "Temperature": 26.0, "Humidity": 75.0, "Rainfall": rf, "Visibility": vs, "Wind_Speed": 12.0,
                "Nearby_POI_Density": 45.0, "Event_Holiday": 0, "Peak_Hour_Indicator": is_peak, "Rush_Hour_Score": (is_peak * 3) + 2,
                "Hour_Sin": np.sin(2 * np.pi * hour / 24.0), "Hour_Cos": np.cos(2 * np.pi * hour / 24.0),
                "Day_Sin": 0.0, "Day_Cos": 1.0, "Month_Sin": 0.0, "Month_Cos": 1.0,
                "Weather_Impact_Score": w_imp, "Weather_x_Hour": w_imp * hour,
                "Lag_1": lag_1, "Lag_2": lag_2, "Lag_3": lag_2, "Lag_4": lag_2, "Lag_6": lag_2, "Lag_12": lag_2,
                "Rolling_Mean_3": (lag_1 + lag_2) / 2.0, "Rolling_Std_3": 5.0, "Rolling_Mean_6": lag_2, "Rolling_Mean_12": lag_2, "Rolling_Mean_24": lag_2, "EMA": lag_1
            }
            df = pd.DataFrame([feats])[stack["feature_cols"]]
            prediction = max(int(stack["meta"].predict(np.column_stack((stack["lgb"].predict(df), stack["xgb"].predict(df))))), 15)
        else: prediction = max(int(((lag_1 + lag_2) / 2.0) * rmod * wmod + (140.0 if is_peak else 0)), 15)
        cap = lanes * 1500
        prediction = min(prediction, cap)
        pct = (prediction / cap) * 100.0
        route = "🚨 High saturation. Divert traffic flow to secondary bypass routes immediately." if pct > 75 else "⚠️ Moderate volume building. Recommend micro-adjusting ramp timers." if pct > 45 else "✅ Traffic flowing smoothly within normal bounds."
        return jsonify({"predicted_vehicle_count": prediction, "pcu_equivalency": round(prediction * 1.15, 1), "surge_ceiling_buffer": max(cap - prediction, 0), "total_capacity": cap, "smart_routing_recommendation": route}), 200
    except Exception as e: return jsonify({"error": str(e)}), 400

if __name__ == '__main__': app.run(host='0.0.0.0', port=5000)
