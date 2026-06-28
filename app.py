from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import numpy as np
import pickle
import os

app = Flask(__name__)

# --- SAFE PRODUCTION ARTIFACT INITIALIZER ---
def load_ml_stack():
    """Safely handles unpickling mechanics within server memory threads"""
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

# --- STANDALONE INTERACTIVE DASHBOARD UI LAYOUT ---
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Intelligent Live Traffic Management Dashboard</title>
    <link href="https://jsdelivr.net" rel="stylesheet">
    <link href="https://googleapis.com" rel="stylesheet">
    <style>
        :root {
            --bg-main: #f8fafc;
            --surface: #ffffff;
            --text-primary: #0f172a;
            --text-secondary: #64748b;
            --primary: #2563eb;
            --primary-hover: #1d4ed8;
            --border: #e2e8f0;
            --accent-bg: #f1f5f9;
        }
        body {
            font-family: 'Inter', sans-serif;
            background-color: var(--bg-main);
            color: var(--text-primary);
            -webkit-font-smoothing: antialiased;
        }
        .navbar {
            background-color: var(--surface);
            border-bottom: 1px solid var(--border);
            padding: 16px 0;
        }
        .navbar-brand {
            font-weight: 700;
            color: var(--text-primary) !important;
            font-size: 1.25rem;
        }
        .card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 16px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
            padding: 24px;
            margin-bottom: 0; /* Managed by grid layout gaps */
        }
        .form-label {
            font-weight: 600;
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: var(--text-secondary);
            margin-bottom: 6px;
        }
        .form-control, .form-select {
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 11px 14px;
            font-size: 0.95rem;
            color: var(--text-primary);
            background-color: #fff;
        }
        .btn-primary {
            background-color: var(--primary);
            border: none;
            border-radius: 10px;
            padding: 14px;
            font-weight: 600;
            font-size: 0.95rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .btn-primary:hover {
            background-color: var(--primary-hover);
        }
        .metric-title {
            font-size: 0.8rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: var(--text-secondary);
            margin-bottom: 8px;
        }
        .metric-value {
            font-size: 2.5rem;
            font-weight: 700;
            color: var(--text-primary);
            line-height: 1;
        }
        .nested-metric {
            background: #f8fafc;
            border-radius: 12px;
            padding: 14px;
            border: 1px solid var(--border);
        }
        .nested-metric-label {
            font-size: 0.8rem;
            font-weight: 600;
            text-transform: uppercase;
            color: var(--text-secondary);
            margin-bottom: 4px;
        }
        .nested-metric-value {
            font-size: 1.2rem;
            font-weight: 700;
        }
        .progress {
            background-color: #f1f5f9;
            height: 8px !important;
            border-radius: 999px;
        }
        .recommendation-box {
            border-left: 4px solid var(--primary);
            background-color: #eff6ff;
            border-radius: 0 12px 12px 0;
            padding: 16px;
        }
    </style>
</head>
<body class="bg-light pb-5">

    <nav class="navbar mb-5">
        <div class="container">
            <span class="navbar-brand text-dark d-flex align-items-center">
                <span class="me-2">🚦</span> Innovexa Catalyst System
            </span>
            <span class="badge bg-success-subtle text-success border border-success-subtle py-2 px-3 fw-semibold rounded-pill">
                ● Connected to Live Engine Stream
            </span>
        </div>
    </nav>

    <div class="container" style="max-width: 1100px;">
        <div class="row g-4">
            
            <!-- Left Grid Panel (Workspace Input Controls) -->
            <div class="col-lg-5">
                <div class="card p-4 h-100">
                    <h5 class="fw-bold mb-4 text-dark">Workspace Parameters</h5>
                    <form id="trafficForm" class="row g-3">
                        <div class="col-12">
                            <label class="form-label">Hour of Day (0-23)</label>
                            <input type="number" class="form-control" name="hour" min="0" max="23" value="10" required>
                        </div>
                        <div class="col-12">
                            <label class="form-label">Road Classification</label>
                            <select class="form-select" name="road_classification">
                                <option value="Express Highway">Express Highway</option>
                                <option value="Arterial Route">Arterial Route</option>
                                <option value="Urban Commuter Route">Urban Commuter Route</option>
                            </select>
                        </div>
                        <div class="col-12">
                            <label class="form-label">Active Lanes</label>
                            <input type="number" class="form-control" name="active_lanes" min="1" max="10" value="4" required>
                        </div>
                        <div class="col-12">
                            <label class="form-label">Weather State</label>
                            <select class="form-select" name="weather_state">
                                <option value="Clear Skies">Clear Skies</option>
                                <option value="Active Rainfall">Active Rainfall</option>
                                <option value="Heavy Storms">Heavy Storms</option>
                            </select>
                        </div>
                        <div class="col-6">
                            <label class="form-label">Lag 1 Volume (T-15m)</label>
                            <input type="number" class="form-control" name="lag_1" value="210" required>
                        </div>
                        <div class="col-6">
                            <label class="form-label">Lag 2 Volume (T-30m)</label>
                            <input type="number" class="form-control" name="lag_2" value="195" required>
                        </div>
                        <div class="col-12 mt-4">
                            <button type="submit" class="btn btn-primary w-100 text-uppercase tracking-wider">Execute Stacking Ensemble</button>
                        </div>
                    </form>
                </div>
            </div>

            <!-- Right Grid Panel (Beautifully Ordered Visual Metrics) -->
            <div class="col-lg-7">
                <div class="d-flex flex-column h-100 gap-4">
                    
                    <!-- Volume Forecast Response Area -->
                    <div class="card p-4 flex-grow-1 d-flex flex-column justify-content-center">
                        <span class="metric-title">Ensemble Forecasted Volume</span>
                        <div class="d-flex align-items-baseline gap-2 mt-2">
                            <div class="metric-value text-primary" id="volOut">0</div>
                            <span class="text-secondary fw-semibold fs-5">vehicles / 15-min</span>
                        </div>
                    </div>

                    <!-- Matrix Component Analytical Grid -->
                    <div class="card p-4">
                        <h6 class="fw-bold mb-3 text-dark">📊 Capacity Analytics Matrix</h6>
                        <div class="row g-3 mb-4">
                            <div class="col-6">
                                <div class="nested-metric">
                                    <div class="nested-metric-label">PCU Equivalency</div>
                                    <div class="nested-metric-value mt-1" id="pcuOut">0</div>
                                </div>
                            </div>
                            <div class="col-6">
                                <div class="nested-metric">
                                    <div class="nested-metric-label">Surge Ceiling Buffer</div>
                                    <div class="nested-metric-value mt-1" id="bufferOut">0</div>
                                </div>
                            </div>
                        </div>
                        
                        <div class="d-flex justify-content-between align-items-center mb-2">
                            <span class="text-secondary small fw-medium">Network Saturation Index</span>
                            <span class="fw-bold small" id="progressPct">0%</span>
                        </div>
                        <div class="progress">
                            <div class="progress-bar bg-primary" id="progressFill" style="width: 0%"></div>
