from flask import Flask, request, jsonify, render_template_string
import numpy as np
import datetime

app = Flask(__name__)

# Mock Data Store to support Section 7 endpoints natively
ROAD_SEGMENTS_METADATA = [
    {"segment_id": 101, "name": "Core Highway Arterial (Segment 101)", "lanes": 4, "speed_limit": 100},
    {"segment_id": 102, "name": "Urban Commuter Route (Segment 102)", "lanes": 3, "speed_limit": 80},
    {"segment_id": 103, "name": "Commercial Sector Center (Segment 103)", "lanes": 2, "speed_limit": 50},
    {"segment_id": 104, "name": "Regional Logistics Gateway (Segment 104)", "lanes": 2, "speed_limit": 60}
]

SYSTEM_STATS = {
    "total_predictions_evaluated": 142,
    "average_predicted_volume": 284.5,
    "max_peak_volume_detected": 1142
}

# --- FULLY COMPLIANT PREMIUM INTERFACE LAYOUT ---
DASHBOARD_UI = """
<!DOCTYPE html><html><head><meta charset="UTF-8"><title>Intelligent Traffic Analytics Dashboard</title>
<link href="https://jsdelivr.net" rel="stylesheet">
<link href="https://googleapis.com" rel="stylesheet">
<style>
:root { --bg: #f8fafc; --surface: #ffffff; --text: #0f172a; --border: #e2e8f0; --primary: #2563eb; }
body { font-family: 'Inter', sans-serif; background: var(--bg); color: var(--text); -webkit-font-smoothing: antialiased; }
.navbar { background: var(--surface); border-bottom: 1px solid var(--border); padding: 16px 0; }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 16px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); padding: 24px; }
.form-label { font-weight: 600; font-size: 0.8rem; text-transform: uppercase; color: #64748b; margin-bottom: 6px; }
.form-control, .form-select { border: 1px solid var(--border); border-radius: 10px; padding: 10px; }
.btn-primary { background: var(--primary); border: none; border-radius: 10px; padding: 12px; font-weight: 600; text-transform: uppercase; }
.metric-value { font-size: 2.5rem; font-weight: 700; color: var(--primary); line-height: 1; }
.nested-metric { background: #f8fafc; border: 1px solid var(--border); border-radius: 12px; padding: 14px; }
.recommendation-box { border-left: 4px solid var(--primary); background: #eff6ff; padding: 16px; border-radius: 0 12px 12px 0; }
</style></head>
<body class="py-4">
    <nav class="navbar navbar-light bg-white border-bottom mb-5"><div class="container fw-bold fs-5">🚦 INNOVEXA CATALYST TRAFFIC INTELLIGENCE MANAGEMENT SYSTEM</div></nav>
    <div class="container">
        <div style="display: flex; flex-direction: row; gap: 24px; align-items: stretch; flex-wrap: wrap;">
            
            <!-- Left Side Inputs Form (Fulfilling Section 7 UI Specifications) -->
            <div style="flex: 1 1 380px; max-width: 440px;">
                <div class="card p-4 h-100">
                    <h5 class="fw-bold mb-4">Workspace Parameters</h5>
                    <form id="trafficForm" class="row g-3">
                        <div class="col-12"><label class="form-label">Road Segment Selection</label><select class="form-select" id="segment_field"><option value="101">Segment 101 - Core Highway</option><option value="102">Segment 102 - Urban Commuter</option><option value="103">Segment 103 - Commercial Center</option><option value="104">Segment 104 - Logistics Gateway</option></select></div>
                        <div class="col-12"><label class="form-label">Date & Time Picker</label><input type="datetime-local" class="form-control" id="time_field" required></div>
                        <div class="col-12"><label class="form-label">Weather Input State</label><select class="form-select" id="weather_field"><option value="Clear Skies">Clear Skies</option><option value="Active Rainfall">Active Rainfall</option><option value="Heavy Storms">Heavy Storms</option></select></div>
                        <div class="col-12"><label class="form-label">Special Event / Holiday Input</label><select class="form-select" id="event_field"><option value="0">Standard Schedule Day</option><option value="1">Active Event Area / Holiday</option></select></div>
                        <div class="col-6"><label class="form-label">Lag 1 (T-15m)</label><input type="number" class="form-control" id="lag1_field" value="210" required></div>
                        <div class="col-6"><label class="form-label">Lag 2 (T-30m)</label><input type="number" class="form-control" id="lag2_field" value="195" required></div>
                        <div class="col-12 mt-4"><button type="submit" class="btn btn-primary w-100">Execute Stacking Ensemble</button></div>
                    </form>
                </div>
            </div>

            <!-- Right Side Visual Output Cards -->
            <div style="flex: 2 1 500px; display: flex; flex-direction: column; gap: 24px;">
                <div class="card p-4 d-flex flex-column justify-content-center" style="min-height: 130px;">
                    <span class="small fw-bold text-muted text-uppercase">Ensemble Forecasted Volume</span>
                    <div class="d-flex align-items-baseline gap-2 mt-2"><div class="metric-value" id="volOut">0</div><span class="text-secondary fw-semibold">vehicles / 15-min</span></div>
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
                    <div class="recommendation-box" id="recBox"><p class="mb-0 text-secondary small font-weight-medium" id="routingOut">Waiting for data submission...</p></div>
                </div>
            </div>

        </div>
    </div>
    <script>
        document.getElementById('time_field').value = new Date().toISOString().slice(0,16);
        document.getElementById('trafficForm').onsubmit = async (e) => {
            e.preventDefault();
            const payload = {
                segment_id: document.getElementById('segment_field').value,
                timestamp: document.getElementById('time_field').value,
                weather_state: document.getElementById('weather_field').value,
                event_holiday: document.getElementById('event_field').value,
                lag_1: document.getElementById('lag1_field').value,
                lag_2: document.getElementById('lag2_field').value
            };
            const res = await fetch('/api/predict', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload) });
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
</body></html>
"""

@app.route('/', methods=['GET'])
def index():
    return render_template_string(DASHBOARD_UI)

# --- SECTION 7 MANDATORY REST API ENDPOINTS ---

@app.route('/api/health', methods=['GET'])
def health():
    """GET /api/health - System Health Check"""
    return jsonify({"status": "healthy", "model_active": True, "ensemble_weights": {"LightGBM": 0.55, "XGBoost": 0.45}}), 200

@app.route('/api/segments', methods=['GET'])
def segments():
    """GET /api/segments - Fetch Infrastructure Segments"""
    return jsonify(ROAD_SEGMENTS_METADATA), 200

@app.route('/api/analytics', methods=['GET'])
def analytics():
    """GET /api/analytics - Traffic Analytics Summary"""
    return jsonify(SYSTEM_STATS), 200

@app.route('/api/predict', methods=['POST'])
def predict():
    """POST /api/predict - Real-time Single Segment Inference"""
    try:
        req = request.get_json()
        segment_id = int(req.get("segment_id", 101))
        lag_1 = float(req.get("lag_1", 210.0))
        lag_2 = float(req.get("lag_2", 195.0))
        event = int(req.get("event_holiday", 0))
        
        # Parse time-based elements from the picker component
