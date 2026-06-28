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

# --- PREMIUM STYLED DASHBOARD UI ---
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Intelligent Traffic Management Dashboard</title>
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
            --border-color: #e2e8f0;
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
        }
        
        body {
            font-family: 'Inter', sans-serif;
            background-color: var(--bg-main);
            color: var(--text-primary);
            -webkit-font-smoothing: antialiased;
        }

        .navbar {
            background-color: var(--surface);
            border-bottom: 1px solid var(--border-color);
            padding: 16px 0;
        }

        .navbar-brand {
            font-weight: 700;
            color: var(--text-primary) !important;
            letter-spacing: -0.5px;
            font-size: 1.25rem;
        }

        .card {
            background: var(--surface);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }

        .form-label {
            font-weight: 600;
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: var(--text-secondary);
            margin-bottom: 6px;
        }

        .form-control, .form-select {
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 11px 14px;
            font-size: 0.95rem;
            color: var(--text-primary);
            background-color: #fff;
            transition: border-color 0.15s ease;
        }

        .form-control:focus, .form-select:focus {
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
        }

        .btn-primary {
            background-color: var(--primary);
            border: none;
            border-radius: 10px;
            padding: 14px;
            font-weight: 600;
            font-size: 0.95rem;
            letter-spacing: 0.3px;
            transition: background-color 0.15s ease;
        }

        .btn-primary:hover {
            background-color: var(--primary-hover);
        }

        .metric-title {
            font-size: 0.8rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: var(--text-secondary);
        }

        .metric-value {
            font-size: 2.5rem;
            font-weight: 700;
            letter-spacing: -1px;
            color: var(--text-primary);
        }

        .nested-metric {
            background: #f8fafc;
            border-radius: 12px;
            padding: 16px;
            border: 1px solid var(--border-color);
        }

        .nested-metric-label {
            font-size: 0.85rem;
            color: var(--text-secondary);
            font-weight: 500;
        }

        .nested-metric-value {
            font-size: 1.15rem;
            font-weight: 700;
            color: var(--text-primary);
        }

        .progress {
            background-color: #f1f5f9;
            border-radius: 999px;
            height: 8px !important;
        }

        .progress-bar {
            background-color: var(--primary);
            border-radius: 999px;
        }

        .recommendation-box {
            border-left: 4px solid var(--primary);
            background-color: #eff6ff;
            border-radius: 0 12px 12px 0;
            padding: 20px;
        }
    </style>
</head>
<body>

    <nav class="navbar mb-5">
        <div class="container">
            <span class="navbar-brand d-flex align-items-center">
                <span class="me-2 fs-4">🚦</span> Innovexa Catalyst System
            </span>
            <span class="badge bg-success-subtle text-success border border-success-subtle py-2 px-3 fw-medium rounded-pill">
                ● Live Engine Stream Active
            </span>
        </div>
    </nav>

    <div class="container mb-5" style="max-width: 1140px;">
        <div class="row g-4">
            
            <!-- Left Panel (Inputs) -->
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
                                <option value="Urban Commuter">Urban Commuter Route</option>
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

            <!-- Right Panel (Outputs) -->
            <div class="col-lg-7">
                <div class="d-flex flex-column h-100 justify-content-between gap-4">
                    
                    <!-- Forecast Card -->
                    <div class="card p-4 flex-grow-1 d-flex flex-column justify-content-center">
                        <span class="metric-title">Ensemble Forecasted Volume</span>
                        <div class="d-flex align-items-baseline gap-2 mt-1">
                            <div class="metric-value" id="volOut">0</div>
                            <span class="text-secondary fw-semibold fs-5">vehicles</span>
                        </div>
                    </div>

                    <!-- Matrix Card -->
                    <div class="card p-4">
                        <h6 class="fw-bold mb-3 d-flex align-items-center text-dark">
                            <span class="me-2 fs-5">📊</span> Capacity Analytics Matrix
                        </h6>
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
                            <span class="text-secondary small fw-medium">Capacity Utilization Threshold</span>
                            <span class="fw-bold small" id="progressPct">0%</span>
                        </div>


