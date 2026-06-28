from flask import Flask, request, jsonify, render_template_string
import numpy as np

app = Flask(__name__)

# Ultra-compact, cache-proof side-by-side dashboard layout layout
UI = """
<!DOCTYPE html><html><head><meta charset="UTF-8"><title>Dashboard</title>
<link href="https://jsdelivr.net" rel="stylesheet">
<link href="https://googleapis.com" rel="stylesheet">
<style>
:root { --bg: #f8fafc; --surface: #ffffff; --text: #0f172a; --border: #e2e8f0; }
body { font-family: 'Inter', sans-serif; background: var(--bg); color: var(--text); -webkit-font-smoothing: antialiased; }
.navbar { background: var(--surface); border-bottom: 1px solid var(--border); padding: 16px 0; }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 16px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); padding: 24px; }
.form-control, .form-select { border: 1px solid var(--border); border-radius: 10px; padding: 10px; }
.btn-primary { background: #2563eb; border: none; border-radius: 10px; padding: 12px; font-weight: 600; text-transform: uppercase; }
.metric-value { font-size: 2.5rem; font-weight: 700; color: #2563eb; line-height: 1; }
.nested-metric { background: #f8fafc; border: 1px solid var(--border); border-radius: 12px; padding: 14px; }
.recommendation-box { border-left: 4px solid #2563eb; background: #eff6ff; padding: 16px; border-radius: 0 12px 12px 0; }
</style></head>
<body class="py-4">
    <nav class="navbar bg-white border-bottom mb-5" style="padding: 16px 0;"><div class="container fw-bold">🚦 Innovexa Catalyst Control Center</div></nav>
    <div class="container">
        <div style="display: flex; flex-direction: row; gap: 24px; align-items: stretch; flex-wrap: wrap;">
            <div style="flex: 1 1 380px; max-width: 440px;">
                <div class="card p-4 h-100">
                    <h5 class="fw-bold mb-4">Workspace Parameters</h5>
                    <form id="trafficForm" class="row g-3">
                        <div class="col-12"><label class="form-label small fw-bold text-muted">HOUR OF DAY (0-23)</label><input type="number" class="form-control" name="hour" min="0" max="23" value="10" required></div>
                        <div class="col-12"><label class="form-label small fw-bold text-muted">ROAD TYPE</label><select class="form-select" name="road_classification"><option value="Express Highway">Express Highway</option><option value="Arterial Route">Arterial Route</option><option value="Urban Commuter Route">Urban Commuter Route</option></select></div>
                        <div class="col-12"><label class="form-label small fw-bold text-muted">ACTIVE LANES</label><input type="number" class="form-control" name="active_lanes" min="1" max="10" value="4" required></div>
                        <div class="col-12"><label class="form-label small fw-bold text-muted">WEATHER STATE</label><select class="form-select" name="weather_state"><option value="Clear Skies">Clear Skies</option><option value="Active Rainfall">Active Rainfall</option><option value="Heavy Storms">Heavy Storms</option></select></div>
                        <div class="col-6"><label class="form-label small fw-bold text-muted">LAG 1 (T-15M)</label><input type="number" class="form-control" name="lag_1" value="210" required></div>
                        <div class="col-6"><label class="form-label small fw-bold text-muted">LAG 2 (T-30M)</label><input type="number" class="form-control" name="lag_2" value="195" required></div>
                        <div class="col-12 mt-4"><button type="submit" class="btn btn-primary w-100">Run Analytics</button></div>
                    </form>
                </div>
            </div>
            <div style="flex: 2 1 500px; display: flex; flex-direction: column; gap: 24px;">
                <div class="card p-4 d-flex flex-column justify-content-center" style="min-height: 130px;">
                    <span class="small fw-bold text-muted text-uppercase">Ensemble Forecasted Volume</span>
                    <div class="d-flex align-items-baseline gap-2 mt-2">
                        <div class="metric-value" id="volOut">0</div>
                        <span class="text-secondary fw-semibold">vehicles / 15-min</span>
                    </div>
                </div>
                <div class="card p-4">
                    <h6 class="fw-bold mb-3 text-dark">📊 Capacity Analytics Matrix</h6>
                    <div class="row g-3 mb-4">
                        <div class="col-6"><div class="nested-metric"><small class="text-muted fw-bold text-uppercase" style="font-size:0.75rem;">Equivalent Car Space</small><div class="fw-bold fs-5 mt-1" id="pcuOut">0</div></div></div>
                        <div class="col-6"><div class="nested-metric"><small class="text-muted fw-bold text-uppercase" style="font-size:0.75rem;">Remaining Road Space</small><div class="fw-bold fs-5 mt-1" id="bufferOut">0</div></div></div>
                    </div>
                    <div class="d-flex justify-content-between align-items-center mb-2"><span class="text-secondary small fw-medium">Road Capacity Used</span><span class="fw-bold small" id="progressPct">0%</span></div>
                    <div class="progress" style="height: 10px;"><div class="progress-bar bg-primary" id="progressFill" style="width:0%"></div></div>
                </div>
                <div class="card p-4">
                    <h6 class="fw-bold mb-3 text-dark">🧠 Smart Routing Recommendation</h6>
                    <div class="recommendation-box" id="recBox"><p class="mb-0 text-secondary small font-weight-medium" id="routingOut">Waiting for workspace execution data submission...</p></div>
                </div>
            </div>
        </div>
    </div>
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
    </script>
</body>
</html>
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
        
        prediction = max(int(((lag_1 + lag_2) / 2.0) * rmod * wmod + (140.0 if is_peak else 0)), 15)
        cap = lanes * 1500
        prediction = min(prediction, cap)
        pct = (prediction / cap) * 100.0
        
        route = "🚨 High saturation. Divert traffic flow to secondary bypass routes immediately." if pct > 75 else "⚠️ Moderate volume building. Recommend micro-adjusting ramp timers." if pct > 45 else "✅ Traffic flowing smoothly within normal bounds."
        return jsonify({"predicted_vehicle_count": prediction, "pcu_equivalency": round(prediction * 1.15, 1), "surge_ceiling_buffer": max(cap - prediction, 0), "total_capacity": cap, "smart_routing_recommendation": route}), 200
    except Exception as e: return jsonify({"error": str(e)}), 400

if __name__ == '__main__': app.run(host='0.0.0.0', port=5000)
